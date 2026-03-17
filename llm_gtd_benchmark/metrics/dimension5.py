"""
Dimension 5 — Fairness & Debiasing Evaluator.

Theoretical framing
-------------------
Synthetic tabular data generators may faithfully replicate not only the
statistical distribution of real data but also its embedded demographic biases.
This dimension audits whether the generated data encodes, amplifies, or
suppresses discriminatory correlations between protected attributes (e.g.,
gender, race) and the prediction target.

Two complementary audit tracks are provided:

    Intrinsic bias audit (NMI)
        Normalized Mutual Information between each protected attribute A and
        the prediction target Y, evaluated on both the real training data
        (reference baseline) and the synthetic data (audit subject):

            NMI(A; Y) = MI(A; Y) / sqrt(H(A) · H(Y))

        NMI ∈ [0, 1] is scale-invariant and cross-dataset comparable.
        Interpretation:
            NMI_synth > NMI_real  →  synthetic data amplifies real-world bias
            NMI_synth < NMI_real  →  synthetic data suppresses bias (may be
                                      over-corrected / distribution-distorted)
            NMI_synth ≈ NMI_real  →  bias structure faithfully preserved

    Downstream disparity audit (TSTR probe)
        A fixed-capacity XGBoost probe is trained on synthetic data (TSTR:
        Train on Synthetic, Test on Real) and its predictions on the real test
        set are stratified by protected attribute group.  Three disparity
        metrics are computed for classification tasks:

            ΔDP   — Demographic Parity Difference (Feldman et al. 2015)
                    max pairwise |P(Ŷ=pos | A=i) − P(Ŷ=pos | A=j)|
                    (macro-OvR for multi-class)

            ΔEOP  — Equal Opportunity Difference (Hardt et al. 2016)
                    max pairwise |TPR_i − TPR_j|  — measures whether the
                    probe rewards/penalises truly positive cases equally
                    across groups.

            ΔEO   — Equalized Odds Difference (Hardt et al. 2016)
                    max(max-pairwise TPR gap, max-pairwise FPR gap)
                    ΔEOP considers only the "benefit" side (TPR); ΔEO adds
                    the "harm" side (FPR) — e.g. a lender that over-approves
                    unqualified applicants from one demographic group.

        For regression tasks, one metric is computed:

            ΔSPG  — Statistical Parity Gap
                    max pairwise |E[Ŷ | A=i] − E[Ŷ | A=j]|
                    Lebesgue-integral generalisation of ΔDP to continuous
                    prediction spaces (Agarwal et al. 2018).

        Multi-class variants use One-vs-Rest (OvR) with macro-averaging.

Engineering guarantees
----------------------
- FairSpec is a pure specification object (no data): protected columns,
  target column, task type, etc.  Analogous to LogicSpec for Dimension 2.
- KBinsDiscretizer is fitted **on real_train_df** in FairnessEvaluator.__init__
  providing consistent bin boundaries across NMI computation and group-
  splitting — the same real-data domain anchor used by other dimensions.
- Feature preprocessing (ColumnTransformer) is fitted on test_real_df in
  evaluate(), following Dimension 3's convention: no real training statistics
  leak into the TSTR branch.
- Groups with < min_group_size samples in the test set emit
  GroupCollapseWarning and return NaN — never Laplace-smoothed.  Smoothing
  at small n produces spuriously low disparity readings that would pass
  FAccT-style audits incorrectly.
- np.nanmax / np.nanmin are used throughout so that collapsed groups do not
  propagate NaN to metrics of valid groups.
- The internal XGBoost probe uses fixed hyperparameters (n_estimators=100,
  max_depth=6) — deliberately decoupled from Dimension 3's Optuna-tuned
  models.  The probe's sole purpose is bias auditing, not maximising
  predictive performance.
- Intersectional fairness (Crenshaw 1989) over the Cartesian product of all
  protected columns is opt-in to avoid combinatorial group collapse.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from llm_gtd_benchmark.core.schema import DataSchema
from llm_gtd_benchmark.metrics.dimension3 import TaskType

logger = logging.getLogger(__name__)

_NAN = float("nan")
# TaskType string values (avoids repeated .value accesses)
_BINARY_VAL = TaskType.BINARY_CLASS.value
_MULTI_VAL = TaskType.MULTI_CLASS.value
_REGRESSION_VAL = TaskType.REGRESSION.value


# ---------------------------------------------------------------------------
# Custom warning
# ---------------------------------------------------------------------------


class GroupCollapseWarning(UserWarning):
    """Emitted when a protected-attribute group has fewer than min_group_size
    samples in the real test set, making disparity metrics statistically
    unreliable.

    Metrics for the affected group are set to NaN rather than Laplace-smoothed
    to avoid producing spurious low-disparity readings that would misrepresent
    the fairness of the evaluated model."""


# ---------------------------------------------------------------------------
# FairSpec — pure specification (no data)
# ---------------------------------------------------------------------------


@dataclass
class FairSpec:
    """Fairness specification for Dimension 5 evaluation.

    Pure specification object — holds no data and performs no computation.
    Analogous to :class:`~llm_gtd_benchmark.core.logic_spec.LogicSpec` for
    Dimension 2.  All data-dependent fitting happens in
    :class:`FairnessEvaluator`.

    Parameters
    ----------
    protected_cols:
        Column names of the protected attributes (e.g., ``["gender", "race"]``).
        May include continuous columns (e.g., ``"age"``); these will be
        automatically discretized using real-data KBinsDiscretizer boundaries.
    target_col:
        Column name of the prediction target.  Should match the ``target_col``
        used in Dimension 3 for interpretable cross-dimension comparison.
    task_type:
        :class:`~llm_gtd_benchmark.metrics.dimension3.TaskType` enum —
        ``BINARY_CLASS``, ``MULTI_CLASS``, or ``REGRESSION``.
    intersectional:
        When ``True``, also compute ΔDP over the Cartesian product of all
        protected attributes (e.g., treating ``"Female×Black"`` as a single
        group).  Opt-in only — combinatorial group collapse risk grows
        steeply with the number of protected columns and their cardinalities.
        Default: ``False``.
    min_group_size:
        Minimum number of test-set samples a protected group must have to be
        included in disparity calculations.  Groups below this threshold receive
        NaN metrics and trigger :class:`GroupCollapseWarning`.  The CLT-derived
        empirical threshold of 30 is the default.  Default: ``30``.
    n_bins_continuous:
        Number of equal-frequency (quantile) bins for KBinsDiscretizer applied
        to continuous protected attributes and continuous targets during NMI
        computation.  Default: ``5``.

    Examples
    --------
    >>> spec = FairSpec(
    ...     protected_cols=["gender", "race"],
    ...     target_col="income",
    ...     task_type=TaskType.BINARY_CLASS,
    ...     intersectional=True,
    ... )
    """

    protected_cols: List[str]
    target_col: str
    task_type: TaskType
    intersectional: bool = False
    min_group_size: int = 30
    n_bins_continuous: int = 5

    def __repr__(self) -> str:
        return (
            f"FairSpec(protected={self.protected_cols}, "
            f"target='{self.target_col}', "
            f"task={self.task_type.value}, "
            f"intersectional={self.intersectional})"
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Dim5Result:
    """Output of :class:`FairnessEvaluator`.

    Metric conventions
    ------------------
    - All disparity metrics (**ΔDP, ΔEOP, ΔEO, ΔSPG**): lower is fairer; 0
      means perfect parity.  Values ∈ [0, 1] for classification; unbounded
      for regression.
    - **NMI**: lower means weaker demographic–target coupling (less bias);
      values ∈ [0, 1].
    - ``NaN`` indicates the metric could not be computed — typically because
      fewer than two valid groups (≥ min_group_size) exist in the test set.

    Attributes
    ----------
    protected_cols:
        Protected attribute column names from :class:`FairSpec`.
    target_col:
        Prediction target column name.
    task_type:
        String value of the :class:`TaskType` (e.g. ``"binary_classification"``).
    bias_nmi_real:
        NMI between each protected attribute and the target on the **real
        training data** (reference baseline).
        ``Dict[protected_col → NMI ∈ [0, 1]]``
    bias_nmi_synth:
        NMI on the **synthetic data** (audit subject).
        ``NMI_synth > NMI_real`` → synthetic data amplifies real-world bias.
    delta_dp:
        Demographic Parity Difference per protected column.
        Max pairwise positive-prediction-rate gap (binary: class 1;
        multi-class: macro-OvR). Classification tasks only.
    delta_eop:
        Equal Opportunity Difference (Hardt et al. 2016) per protected column.
        Max pairwise |TPR_i − TPR_j|. Classification only; ``None`` for
        regression.
    delta_eo:
        Equalized Odds Difference (Hardt et al. 2016) per protected column.
        ``max(max-pairwise TPR gap, max-pairwise FPR gap)``. Classification
        only; ``None`` for regression.
    stat_parity_gap:
        Statistical Parity Gap per protected column.
        Max pairwise |E[Ŷ | A=i] − E[Ŷ | A=j]|. Regression only; ``None``
        for classification.
    intersectional_delta_dp:
        ΔDP (or ΔSPG for regression) over the Cartesian product of all
        protected attributes.  Scalar.  Populated only when
        ``FairSpec.intersectional=True``; otherwise ``None``.
    group_collapse_warnings:
        Human-readable strings identifying each group excluded from disparity
        computation due to insufficient test-set sample counts.
    """

    protected_cols: List[str]
    target_col: str
    task_type: str

    # Intrinsic bias (NMI)
    bias_nmi_real: Dict[str, float] = field(default_factory=dict)
    bias_nmi_synth: Dict[str, float] = field(default_factory=dict)

    # Classification disparity metrics
    delta_dp: Dict[str, float] = field(default_factory=dict)
    delta_eop: Optional[Dict[str, float]] = None
    delta_eo: Optional[Dict[str, float]] = None

    # Regression disparity metric
    stat_parity_gap: Optional[Dict[str, float]] = None

    # Intersectional (opt-in)
    intersectional_delta_dp: Optional[float] = None

    # Diagnostics
    group_collapse_warnings: List[str] = field(default_factory=list)

    # Bootstrap CIs (populated only when n_boot > 0 is passed to evaluate())
    # Dict[protected_col → (lo, hi)] for the primary disparity metrics.
    delta_eo_ci: Optional[Dict[str, Tuple[float, float]]] = None
    delta_dp_ci: Optional[Dict[str, Tuple[float, float]]] = None

    @property
    def summary(self) -> str:
        def _fmt(v: float) -> str:
            return f"{v:.4f}" if not np.isnan(v) else "  N/A "

        def _delta_tag(delta: float) -> str:
            if np.isnan(delta):
                return ""
            if delta > 0:
                return "(↑ amplified)"
            if delta < 0:
                return "(↓ suppressed)"
            return "(= unchanged)"

        is_cls = self.task_type in (_BINARY_VAL, _MULTI_VAL)

        lines = [
            "── Dimension 5: Fairness & Debiasing ─────────────────────────────",
            f"  Target    : {self.target_col}",
            f"  Task      : {self.task_type}",
            f"  Protected : {', '.join(self.protected_cols)}",
            "",
            "  Intrinsic Bias — NMI(A; Y)  [↓ less bias | 0 = independent]",
            f"  {'Attribute':<24} {'NMI_real (ref)':<18} {'NMI_synth':<18} {'Δ(synth−real)'}",
            "  " + "─" * 72,
        ]
        for col in self.protected_cols:
            nmi_r = self.bias_nmi_real.get(col, _NAN)
            nmi_s = self.bias_nmi_synth.get(col, _NAN)
            delta = (nmi_s - nmi_r) if not (np.isnan(nmi_r) or np.isnan(nmi_s)) else _NAN
            tag = _delta_tag(delta)
            lines.append(
                f"  {col:<24} {_fmt(nmi_r):<18} {_fmt(nmi_s):<18} "
                f"{_fmt(delta):<10} {tag}"
            )

        if is_cls and self.delta_dp:
            lines += [
                "",
                "  Downstream Disparity — Classification  [↓ fairer | 0 = perfect parity]",
                f"  {'Attribute':<24} {'ΔDP':<12} {'ΔEOP':<12} {'ΔEO'}",
                "  " + "─" * 56,
            ]
            for col in self.protected_cols:
                dp = self.delta_dp.get(col, _NAN)
                eop = self.delta_eop.get(col, _NAN) if self.delta_eop else _NAN
                eo = self.delta_eo.get(col, _NAN) if self.delta_eo else _NAN
                lines.append(f"  {col:<24} {_fmt(dp):<12} {_fmt(eop):<12} {_fmt(eo)}")

        if self.stat_parity_gap:
            lines += [
                "",
                "  Downstream Disparity — Regression  [↓ fairer | 0 = perfect parity]",
                f"  {'Attribute':<24} {'ΔSPG (mean pred. gap)'}",
                "  " + "─" * 48,
            ]
            for col in self.protected_cols:
                spg = self.stat_parity_gap.get(col, _NAN)
                lines.append(f"  {col:<24} {_fmt(spg)}")

        if self.intersectional_delta_dp is not None:
            label = "Intersectional ΔSPG" if not is_cls else "Intersectional ΔDP"
            lines += ["", f"  {label} : {_fmt(self.intersectional_delta_dp)}"]

        if self.group_collapse_warnings:
            lines += [
                "",
                f"  GroupCollapseWarning — {len(self.group_collapse_warnings)} group(s) skipped:",
            ]
            for w in self.group_collapse_warnings[:5]:
                lines.append(f"    · {w}")
            if len(self.group_collapse_warnings) > 5:
                lines.append(
                    f"    … and {len(self.group_collapse_warnings) - 5} more"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _max_pairwise_diff(values: Dict[Any, float]) -> float:
    """Return max|a − b| over all pairs of non-NaN values.

    For k values this equals ``max − min``, which is O(k) instead of O(k²).
    Returns NaN when fewer than two valid (non-NaN) values exist.
    """
    vals = np.array(list(values.values()), dtype=float)
    valid = vals[~np.isnan(vals)]
    if len(valid) < 2:
        return _NAN
    return float(np.max(valid) - np.min(valid))


# ---------------------------------------------------------------------------
# FairnessEvaluator
# ---------------------------------------------------------------------------


class FairnessEvaluator:
    """Dimension 5 fairness evaluator: intrinsic bias (NMI) + downstream
    disparity (TSTR probe).

    Parameters
    ----------
    schema:
        :class:`~llm_gtd_benchmark.core.schema.DataSchema` built from real
        training data.
    fair_spec:
        :class:`FairSpec` specifying protected columns, target column, task
        type, and evaluation options.
    real_train_df:
        Real training DataFrame.  Used to:

        (a) fit :class:`~sklearn.preprocessing.KBinsDiscretizer` for
            continuous protected attributes and the target (consistent bin
            boundaries for NMI and group splitting);
        (b) compute ``bias_nmi_real`` — the reference bias baseline.

    random_state:
        Global seed for the internal XGBoost probe.  Default: ``42``.

    Examples
    --------
    >>> spec = FairSpec(["gender", "race"], "income", TaskType.BINARY_CLASS)
    >>> evaluator = FairnessEvaluator(schema, spec, train_real_df)
    >>> result5 = evaluator.evaluate(synth_df, test_real_df)
    >>> print(result5.summary)
    """

    def __init__(
        self,
        schema: DataSchema,
        fair_spec: FairSpec,
        real_train_df: pd.DataFrame,
        random_state: int = 42,
    ) -> None:
        self.schema = schema
        self.spec = fair_spec
        self.real_train_df = real_train_df.reset_index(drop=True)
        self.random_state = random_state

        self._is_cls = fair_spec.task_type in (TaskType.BINARY_CLASS, TaskType.MULTI_CLASS)
        self._is_binary = fair_spec.task_type == TaskType.BINARY_CLASS

        # Validate columns exist in real_train_df
        all_required = list(dict.fromkeys(fair_spec.protected_cols + [fair_spec.target_col]))
        missing = [c for c in all_required if c not in real_train_df.columns]
        if missing:
            raise ValueError(
                f"FairnessEvaluator: columns not found in real_train_df: {missing}"
            )

        # Fit KBinsDiscretizer for each continuous protected col and target.
        self._discretizers: Dict[str, KBinsDiscretizer] = {}
        self._fit_discretizers()

        logger.info(
            "FairnessEvaluator ready: protected=%s, target='%s', task=%s",
            fair_spec.protected_cols,
            fair_spec.target_col,
            fair_spec.task_type.value,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        synth_df: pd.DataFrame,
        test_real_df: pd.DataFrame,
        n_boot: int = 0,
        boot_ci: float = 0.95,
    ) -> Dim5Result:
        """Compute all Dimension 5 fairness metrics.

        Parameters
        ----------
        synth_df:
            Synthetic training data (Dim0-cleaned output recommended).
            Must contain ``protected_cols`` and ``target_col``.
        test_real_df:
            Real held-out test data.  Must contain ``protected_cols`` and
            ``target_col`` (true labels for disparity computation).
        n_boot:
            Number of bootstrap replicates for sampling-uncertainty CIs.
            ``0`` (default) disables bootstrapping.  Recommended: 1000.
        boot_ci:
            Confidence level for bootstrap CIs.  Default: 0.95.

        Returns
        -------
        :class:`Dim5Result`
        """
        synth_df = synth_df.reset_index(drop=True)
        test_real_df = test_real_df.reset_index(drop=True)
        group_collapse_warnings: List[str] = []

        # ── Step 1: Intrinsic bias (NMI) ────────────────────────────────────
        logger.info("Dim5: computing NMI on real training data (reference)...")
        bias_nmi_real = self._compute_nmi(self.real_train_df)

        logger.info("Dim5: computing NMI on synthetic data...")
        bias_nmi_synth = self._compute_nmi(synth_df)

        # ── Step 2: Feature encoding ─────────────────────────────────────────
        feature_cols = [c for c in test_real_df.columns if c != self.spec.target_col]
        ct = self._build_feature_pipeline(feature_cols)

        # Fit preprocessing on test_real_df — domain anchor (no leakage of
        # real training statistics into the TSTR synthetic-training branch).
        try:
            ct.fit(test_real_df[feature_cols])
            X_synth_raw = ct.transform(synth_df[feature_cols])
            X_test_raw = ct.transform(test_real_df[feature_cols])
        except Exception as exc:
            logger.error("Dim5: feature encoding failed: %s", exc)
            return Dim5Result(
                protected_cols=self.spec.protected_cols,
                target_col=self.spec.target_col,
                task_type=self.spec.task_type.value,
                bias_nmi_real=bias_nmi_real,
                bias_nmi_synth=bias_nmi_synth,
                group_collapse_warnings=[
                    f"Feature encoding failed ({exc}); disparity metrics not computed."
                ],
            )

        # ── Step 3: Target encoding ──────────────────────────────────────────
        valid_synth = synth_df[self.spec.target_col].notna().values
        valid_test = test_real_df[self.spec.target_col].notna().values
        X_synth_v = X_synth_raw[valid_synth]
        X_test_v = X_test_raw[valid_test]

        y_synth_raw = synth_df[self.spec.target_col][valid_synth]
        y_test_raw = test_real_df[self.spec.target_col][valid_test]

        label_enc: Optional[LabelEncoder] = None
        n_classes = 1

        if self._is_cls:
            label_enc = LabelEncoder()
            all_labels = pd.concat([
                synth_df[self.spec.target_col],
                test_real_df[self.spec.target_col],
            ]).dropna().astype(str)
            label_enc.fit(all_labels)
            n_classes = len(label_enc.classes_)
            y_synth = label_enc.transform(y_synth_raw.astype(str))
            y_test = label_enc.transform(y_test_raw.astype(str))
        else:
            y_synth = y_synth_raw.astype(float).values
            y_test = y_test_raw.astype(float).values

        # Degenerate training check
        if self._is_cls and len(np.unique(y_synth)) < 2:
            logger.warning(
                "Dim5: synthetic data has fewer than 2 distinct classes; "
                "disparity metrics will be unreliable."
            )

        # ── Step 4: Train TSTR probe ─────────────────────────────────────────
        probe = self._build_probe(multiclass=not self._is_binary and self._is_cls)
        logger.info(
            "Dim5: training %s probe on synthetic data (%d rows)...",
            type(probe).__name__,
            X_synth_v.shape[0],
        )
        try:
            probe.fit(X_synth_v, y_synth)
        except Exception as exc:
            logger.error("Dim5: probe training failed: %s", exc)
            return Dim5Result(
                protected_cols=self.spec.protected_cols,
                target_col=self.spec.target_col,
                task_type=self.spec.task_type.value,
                bias_nmi_real=bias_nmi_real,
                bias_nmi_synth=bias_nmi_synth,
                group_collapse_warnings=[
                    f"Probe training failed ({exc}); disparity metrics not computed."
                ],
            )

        # Get predictions on real test set
        try:
            y_pred = probe.predict(X_test_v)
        except Exception as exc:
            logger.error("Dim5: probe prediction failed: %s", exc)
            return Dim5Result(
                protected_cols=self.spec.protected_cols,
                target_col=self.spec.target_col,
                task_type=self.spec.task_type.value,
                bias_nmi_real=bias_nmi_real,
                bias_nmi_synth=bias_nmi_synth,
                group_collapse_warnings=[
                    f"Probe prediction failed ({exc}); disparity metrics not computed."
                ],
            )

        # test_real_df rows aligned with y_test / y_pred
        test_df_valid = test_real_df[valid_test].reset_index(drop=True)

        # ── Step 5: Per-column disparity metrics ─────────────────────────────
        delta_dp: Dict[str, float] = {}
        delta_eop: Optional[Dict[str, float]] = {} if self._is_cls else None
        delta_eo: Optional[Dict[str, float]] = {} if self._is_cls else None
        stat_parity_gap: Optional[Dict[str, float]] = None if self._is_cls else {}

        for prot_col in self.spec.protected_cols:
            logger.info("Dim5: computing disparity metrics for '%s'...", prot_col)
            dp, eop, eo, spg, col_warns = self._disparity_for_column(
                prot_col=prot_col,
                y_true=y_test,
                y_pred=y_pred,
                n_classes=n_classes,
                test_df=test_df_valid,
            )
            group_collapse_warnings.extend(col_warns)

            if self._is_cls:
                delta_dp[prot_col] = dp
                delta_eop[prot_col] = eop  # type: ignore[index]
                delta_eo[prot_col] = eo  # type: ignore[index]
            else:
                stat_parity_gap[prot_col] = spg  # type: ignore[index]

        # ── Step 6: Intersectional metrics (opt-in) ──────────────────────────
        intersectional_delta_dp: Optional[float] = None
        if self.spec.intersectional and len(self.spec.protected_cols) >= 2:
            logger.info("Dim5: computing intersectional disparity...")
            intersectional_delta_dp, inter_warns = self._intersectional_disparity(
                y_true=y_test,
                y_pred=y_pred,
                n_classes=n_classes,
                test_df=test_df_valid,
            )
            group_collapse_warnings.extend(inter_warns)
        elif self.spec.intersectional and len(self.spec.protected_cols) < 2:
            logger.warning(
                "Dim5: intersectional=True but only %d protected column(s) provided; "
                "intersectional ΔDP requires at least 2.",
                len(self.spec.protected_cols),
            )

        result = Dim5Result(
            protected_cols=self.spec.protected_cols,
            target_col=self.spec.target_col,
            task_type=self.spec.task_type.value,
            bias_nmi_real=bias_nmi_real,
            bias_nmi_synth=bias_nmi_synth,
            delta_dp=delta_dp,
            delta_eop=delta_eop,
            delta_eo=delta_eo,
            stat_parity_gap=stat_parity_gap,
            intersectional_delta_dp=intersectional_delta_dp,
            group_collapse_warnings=group_collapse_warnings,
        )

        # ── Bootstrap CIs (opt-in) ────────────────────────────────────────────
        if n_boot > 0 and self._is_cls:
            logger.info("Dim5: bootstrapping CIs (n_boot=%d, ci=%.2f)...", n_boot, boot_ci)
            from llm_gtd_benchmark.utils.bootstrap import bootstrap_ci

            rng = np.random.default_rng(self.random_state)
            n_test = len(test_df_valid)

            delta_eo_ci: Dict[str, Tuple[float, float]] = {}
            delta_dp_ci: Dict[str, Tuple[float, float]] = {}

            for prot_col in self.spec.protected_cols:
                # Resample row indices of (y_true, y_pred, test_df_valid) triplet
                def _boot_eo(idx: np.ndarray, col: str = prot_col) -> float:
                    sub_df = test_df_valid.iloc[idx].reset_index(drop=True)
                    sub_yt = y_test[idx]
                    sub_yp = y_pred[idx]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", GroupCollapseWarning)
                        dp_, eop_, eo_, spg_, _ = self._disparity_for_column(
                            prot_col=col,
                            y_true=sub_yt,
                            y_pred=sub_yp,
                            n_classes=n_classes,
                            test_df=sub_df,
                        )
                    return eo_

                def _boot_dp(idx: np.ndarray, col: str = prot_col) -> float:
                    sub_df = test_df_valid.iloc[idx].reset_index(drop=True)
                    sub_yt = y_test[idx]
                    sub_yp = y_pred[idx]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", GroupCollapseWarning)
                        dp_, eop_, eo_, spg_, _ = self._disparity_for_column(
                            prot_col=col,
                            y_true=sub_yt,
                            y_pred=sub_yp,
                            n_classes=n_classes,
                            test_df=sub_df,
                        )
                    return dp_

                idx_population = np.arange(n_test)
                try:
                    ci_eo = bootstrap_ci(idx_population, _boot_eo, n_boot, boot_ci, rng)
                    delta_eo_ci[prot_col] = ci_eo
                except Exception as exc:
                    logger.warning("Dim5 bootstrap EO CI failed for '%s': %s", prot_col, exc)

                try:
                    ci_dp = bootstrap_ci(idx_population, _boot_dp, n_boot, boot_ci, rng)
                    delta_dp_ci[prot_col] = ci_dp
                except Exception as exc:
                    logger.warning("Dim5 bootstrap DP CI failed for '%s': %s", prot_col, exc)

            result.delta_eo_ci = delta_eo_ci if delta_eo_ci else None
            result.delta_dp_ci = delta_dp_ci if delta_dp_ci else None

        return result

    # ── Internal: discretizer fitting ────────────────────────────────────────

    def _fit_discretizers(self) -> None:
        """Fit KBinsDiscretizer on real_train_df for every continuous column
        among the protected attributes and the target column.

        Both NMI computation and group-splitting use the same fitted instance,
        guaranteeing consistent bin boundaries across both operations.
        """
        cols_to_check = list(
            dict.fromkeys(self.spec.protected_cols + [self.spec.target_col])
        )
        for col in cols_to_check:
            try:
                col_schema = self.schema[col]
            except KeyError:
                logger.warning("Dim5: column '%s' not in schema; no discretizer fitted.", col)
                continue

            if col_schema.col_type != "continuous":
                continue

            n_bins = self.spec.n_bins_continuous
            values = self.real_train_df[col].dropna().values.reshape(-1, 1)

            if len(values) < n_bins:
                n_bins = max(2, len(values))
                logger.warning(
                    "Dim5: column '%s' has only %d non-null values; "
                    "clamping n_bins to %d.",
                    col, len(values), n_bins,
                )

            kbd = KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy="quantile",
                subsample=None,
            )
            kbd.fit(values)
            self._discretizers[col] = kbd
            logger.debug("Dim5: KBinsDiscretizer fitted for continuous col '%s'.", col)

    # ── Internal: discretization ──────────────────────────────────────────────

    def _discretize_col(self, series: pd.Series, col: str) -> np.ndarray:
        """Return a string-encoded array for use in NMI and group splitting.

        - Continuous columns (col in self._discretizers): transform via the
          fitted KBinsDiscretizer; return bin index as a string (e.g. ``"2"``).
        - Categorical columns: return original values cast to str.

        NaN values must be removed by the caller before invoking this method.
        """
        if col in self._discretizers:
            vals = series.values.astype(float).reshape(-1, 1)
            return self._discretizers[col].transform(vals).ravel().astype(int).astype(str)
        return series.astype(str).values

    # ── Internal: NMI computation ─────────────────────────────────────────────

    def _compute_nmi(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute NMI(protected_col, target_col) for every protected attribute.

        Only rows where both columns are non-null contribute to the score.
        Returns NaN for a column when fewer than 10 jointly-valid rows exist.
        """
        target_col = self.spec.target_col
        result: Dict[str, float] = {}

        if target_col not in df.columns:
            logger.warning(
                "Dim5 NMI: target_col '%s' not in df; returning NaN for all.", target_col
            )
            return {col: _NAN for col in self.spec.protected_cols}

        for col in self.spec.protected_cols:
            if col not in df.columns:
                logger.warning("Dim5 NMI: protected_col '%s' not in df; NaN.", col)
                result[col] = _NAN
                continue

            valid_mask = df[col].notna() & df[target_col].notna()
            n_valid = valid_mask.sum()

            if n_valid < 10:
                logger.warning(
                    "Dim5 NMI: only %d jointly valid rows for '%s'; returning NaN.",
                    n_valid,
                    col,
                )
                result[col] = _NAN
                continue

            a_disc = self._discretize_col(df.loc[valid_mask, col], col)
            y_disc = self._discretize_col(df.loc[valid_mask, target_col], target_col)

            try:
                nmi_val = float(
                    normalized_mutual_info_score(a_disc, y_disc, average_method="arithmetic")
                )
            except Exception as exc:
                logger.warning("Dim5 NMI: computation failed for '%s': %s", col, exc)
                nmi_val = _NAN

            result[col] = nmi_val

        return result

    # ── Internal: feature pipeline ────────────────────────────────────────────

    def _build_feature_pipeline(self, feature_cols: List[str]) -> ColumnTransformer:
        """Build a ColumnTransformer for probe training/evaluation features.

        Columns absent from the schema are silently dropped (remainder='drop').
        Robust to synthetic–real column mismatches via OrdinalEncoder
        ``unknown_value=-1``.
        """
        cont_cols: List[str] = []
        cat_cols: List[str] = []

        for c in feature_cols:
            try:
                col_type = self.schema[c].col_type
            except KeyError:
                logger.warning(
                    "Dim5: column '%s' not in schema; excluded from probe features.", c
                )
                continue
            if col_type == "continuous":
                cont_cols.append(c)
            else:
                cat_cols.append(c)

        transformers: List[Tuple] = []
        if cont_cols:
            num_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ])
            transformers.append(("num", num_pipe, cont_cols))
        if cat_cols:
            cat_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
            ])
            transformers.append(("cat", cat_pipe, cat_cols))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    # ── Internal: probe construction ──────────────────────────────────────────

    def _build_probe(self, multiclass: bool = False) -> Any:
        """Build a fixed-capacity XGBoost probe (falls back to RandomForest).

        Fixed hyperparameters (n_estimators=100, max_depth=6) are intentional:
        the probe's purpose is bias auditing, not maximising predictive
        performance.  Decoupled from Dimension 3's Optuna-tuned models.
        """
        rs = self.random_state
        if self._is_cls:
            try:
                from xgboost import XGBClassifier

                obj = "multi:softprob" if multiclass else "binary:logistic"
                eval_m = "mlogloss" if multiclass else "logloss"
                return XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=rs,
                    verbosity=0,
                    objective=obj,
                    eval_metric=eval_m,
                    n_jobs=-1,
                )
            except ImportError:
                from sklearn.ensemble import RandomForestClassifier

                logger.warning(
                    "Dim5: xgboost not installed; using RandomForestClassifier as probe. "
                    "Install xgboost for best results: pip install xgboost"
                )
                return RandomForestClassifier(
                    n_estimators=100, max_depth=6, random_state=rs, n_jobs=-1
                )
        else:
            try:
                from xgboost import XGBRegressor

                return XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=rs,
                    verbosity=0,
                    n_jobs=-1,
                )
            except ImportError:
                from sklearn.ensemble import RandomForestRegressor

                logger.warning(
                    "Dim5: xgboost not installed; using RandomForestRegressor as probe."
                )
                return RandomForestRegressor(
                    n_estimators=100, max_depth=6, random_state=rs, n_jobs=-1
                )

    # ── Internal: per-column disparity ────────────────────────────────────────

    def _disparity_for_column(
        self,
        prot_col: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_classes: int,
        test_df: pd.DataFrame,
    ) -> Tuple[float, float, float, float, List[str]]:
        """Compute disparity metrics for one protected attribute column.

        Returns
        -------
        Tuple of (delta_dp, delta_eop, delta_eo, stat_parity_gap, warnings).

        For classification: delta_dp/eop/eo are populated; stat_parity_gap=NaN.
        For regression: stat_parity_gap is populated; dp/eop/eo=NaN.
        """
        col_warns: List[str] = []

        if prot_col not in test_df.columns:
            col_warns.append(f"'{prot_col}' not in test_df; skipped.")
            return _NAN, _NAN, _NAN, _NAN, col_warns

        # Filter rows where prot_col is NaN so that NaN does not create a
        # spurious group ("nan" string for categoricals, or a corrupted
        # large-negative-integer bin for continuous columns).
        prot_valid = test_df[prot_col].notna().values
        if not prot_valid.any():
            col_warns.append(f"'{prot_col}' is all-NaN in test_df; skipped.")
            return _NAN, _NAN, _NAN, _NAN, col_warns
        test_df_f = test_df[prot_valid].reset_index(drop=True)
        y_true_f = y_true[prot_valid]
        y_pred_f = y_pred[prot_valid]

        group_vals = self._discretize_col(test_df_f[prot_col], prot_col)
        valid_groups, group_masks = self._filter_groups(
            group_vals, prot_col, col_warns
        )

        if len(valid_groups) < 2:
            logger.warning(
                "Dim5: fewer than 2 valid groups for '%s'; all disparity metrics = NaN.",
                prot_col,
            )
            return _NAN, _NAN, _NAN, _NAN, col_warns

        if self._is_cls:
            dp = self._delta_dp_cls(valid_groups, group_masks, y_pred_f, n_classes)
            eop = self._delta_eop(valid_groups, group_masks, y_true_f, y_pred_f, n_classes)
            eo = self._delta_eo(valid_groups, group_masks, y_true_f, y_pred_f, n_classes)
            return dp, eop, eo, _NAN, col_warns
        else:
            spg = self._stat_parity_gap(valid_groups, group_masks, y_pred_f)
            return _NAN, _NAN, _NAN, spg, col_warns

    def _filter_groups(
        self,
        group_vals: np.ndarray,
        prot_col: str,
        col_warns: List[str],
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """Apply min_group_size filter; emit GroupCollapseWarning for small groups."""
        valid_groups: List[str] = []
        group_masks: Dict[str, np.ndarray] = {}

        for g in np.unique(group_vals):
            mask = group_vals == g
            n = int(mask.sum())
            if n < self.spec.min_group_size:
                msg = (
                    f"'{prot_col}'='{g}': {n} samples < "
                    f"min_group_size={self.spec.min_group_size}; excluded (NaN)."
                )
                col_warns.append(msg)
                warnings.warn(msg, GroupCollapseWarning, stacklevel=4)
                logger.warning("Dim5: GroupCollapse — %s", msg)
            else:
                valid_groups.append(str(g))
                group_masks[str(g)] = mask

        return valid_groups, group_masks

    # ── Internal: classification disparity ────────────────────────────────────

    def _delta_dp_cls(
        self,
        valid_groups: List[str],
        group_masks: Dict[str, np.ndarray],
        y_pred: np.ndarray,
        n_classes: int,
    ) -> float:
        """Demographic Parity Difference for classification.

        Binary: P(Ŷ = 1 | A = g), max pairwise gap.
        Multi-class: for each class c, P(Ŷ = c | A = g), max pairwise gap,
        then macro-averaged over classes.
        """
        if self._is_binary:
            rates = {g: float(np.mean(y_pred[group_masks[g]] == 1)) for g in valid_groups}
            return _max_pairwise_diff(rates)

        # Multi-class macro-OvR
        per_class: List[float] = []
        for c in range(n_classes):
            rates = {
                g: float(np.mean(y_pred[group_masks[g]] == c))
                for g in valid_groups
            }
            per_class.append(_max_pairwise_diff(rates))
        valid_vals = [v for v in per_class if not np.isnan(v)]
        return float(np.mean(valid_vals)) if valid_vals else _NAN

    def _delta_eop(
        self,
        valid_groups: List[str],
        group_masks: Dict[str, np.ndarray],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_classes: int,
    ) -> float:
        """Equal Opportunity Difference: max pairwise |TPR_i − TPR_j|.

        Only the True Positive Rate (TPR) side is measured.  For multi-class:
        macro-OvR average.  See delta_eo for the full Equalized Odds metric
        that additionally includes False Positive Rate parity.
        """
        if self._is_binary:
            tprs: Dict[str, float] = {}
            for g in valid_groups:
                mask = group_masks[g]
                pos_mask = y_true[mask] == 1
                tprs[g] = (
                    float(np.mean(y_pred[mask][pos_mask] == 1))
                    if pos_mask.sum() > 0
                    else _NAN
                )
            return _max_pairwise_diff(tprs)

        per_class: List[float] = []
        for c in range(n_classes):
            tprs = {}
            for g in valid_groups:
                mask = group_masks[g]
                pos_mask = y_true[mask] == c
                tprs[g] = (
                    float(np.mean(y_pred[mask][pos_mask] == c))
                    if pos_mask.sum() > 0
                    else _NAN
                )
            per_class.append(_max_pairwise_diff(tprs))
        valid_vals = [v for v in per_class if not np.isnan(v)]
        return float(np.mean(valid_vals)) if valid_vals else _NAN

    def _delta_eo(
        self,
        valid_groups: List[str],
        group_masks: Dict[str, np.ndarray],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_classes: int,
    ) -> float:
        """Equalized Odds Difference (Hardt et al. 2016):
        max(max-pairwise TPR gap, max-pairwise FPR gap).

        ΔEOP only enforces TPR equality (benefit side). ΔEO additionally
        enforces FPR equality (harm side), e.g. protecting against a model
        that over-predicts positive outcomes for one demographic group.
        Multi-class: macro-OvR average of per-class ΔEO.
        """
        def _tpr_fpr(
            c_pos: int,
            g: str,
        ) -> Tuple[float, float]:
            mask = group_masks[g]
            pos_mask = y_true[mask] == c_pos
            neg_mask = ~pos_mask
            tpr = (
                float(np.mean(y_pred[mask][pos_mask] == c_pos))
                if pos_mask.sum() > 0
                else _NAN
            )
            fpr = (
                float(np.mean(y_pred[mask][neg_mask] == c_pos))
                if neg_mask.sum() > 0
                else _NAN
            )
            return tpr, fpr

        def _eo_one_class(c: int) -> float:
            tprs: Dict[str, float] = {}
            fprs: Dict[str, float] = {}
            for g in valid_groups:
                tprs[g], fprs[g] = _tpr_fpr(c, g)
            d_tpr = _max_pairwise_diff(tprs)
            d_fpr = _max_pairwise_diff(fprs)
            if np.isnan(d_tpr) and np.isnan(d_fpr):
                return _NAN
            return float(np.nanmax([d_tpr, d_fpr]))

        if self._is_binary:
            return _eo_one_class(1)

        per_class = [_eo_one_class(c) for c in range(n_classes)]
        valid_vals = [v for v in per_class if not np.isnan(v)]
        return float(np.mean(valid_vals)) if valid_vals else _NAN

    def _stat_parity_gap(
        self,
        valid_groups: List[str],
        group_masks: Dict[str, np.ndarray],
        y_pred: np.ndarray,
    ) -> float:
        """Statistical Parity Gap for regression tasks.

        ΔSPG = max_ij |E[Ŷ | A=i] − E[Ŷ | A=j]|
        = max − min of per-group mean predictions.
        """
        means = {g: float(np.nanmean(y_pred[group_masks[g]])) for g in valid_groups}
        return _max_pairwise_diff(means)

    # ── Internal: intersectional disparity ────────────────────────────────────

    def _intersectional_disparity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_classes: int,
        test_df: pd.DataFrame,
    ) -> Tuple[float, List[str]]:
        """Compute ΔDP (or ΔSPG for regression) over the Cartesian product
        of all protected attributes.

        Group identifiers are formed by joining per-column discretized values
        with ``"×"`` (e.g. ``"Female×Black"``).  Groups below min_group_size
        are excluded with GroupCollapseWarning.
        """
        col_warns: List[str] = []

        # Build per-column discretized arrays
        disc_cols: List[np.ndarray] = []
        for col in self.spec.protected_cols:
            if col not in test_df.columns:
                col_warns.append(
                    f"Intersectional: '{col}' not in test_df; skipping intersectional."
                )
                return _NAN, col_warns
            disc_cols.append(self._discretize_col(test_df[col], col))

        # Vectorised join into compound group labels
        compound_vals = np.array(
            ["×".join(vals) for vals in zip(*disc_cols)],
            dtype=object,
        )

        valid_groups, group_masks = self._filter_groups(
            compound_vals, "intersectional", col_warns
        )

        if len(valid_groups) < 2:
            logger.warning(
                "Dim5 intersectional: fewer than 2 valid compound groups; ΔDP = NaN."
            )
            return _NAN, col_warns

        if self._is_cls:
            dp = self._delta_dp_cls(valid_groups, group_masks, y_pred, n_classes)
        else:
            dp = self._stat_parity_gap(valid_groups, group_masks, y_pred)

        return dp, col_warns

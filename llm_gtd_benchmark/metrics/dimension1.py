"""
Dimension 1 — Distributional Fidelity & Diversity Evaluator.

Theoretical framing
-------------------
Distributional fidelity is assessed at four complementary statistical scales,
forming a closed measurement loop:

    Low-order (marginal)
        KS statistic (continuous) and Total Variation Distance (categorical)
        capture per-column distributional shift — cheap and interpretable.

    Mid-order (pairwise)
        Pearson correlation matrix error (continuous) and Cramér's V matrix
        error (categorical) detect whether the model preserves inter-feature
        dependencies.

    High-order (manifold, Alaa et al. 2022)
        α-precision: fraction of synthetic points that fall inside the support
            of the real distribution (fidelity / no hallucination in latent space).
        β-recall: fraction of real support covered by synthetic points
            (diversity / no mode collapse).
        Both are computed using adaptive k-NN radii, avoiding the need for
        a fixed bandwidth hyper-parameter.

    Global (classifier two-sample test, C2ST)
        XGBoost + RandomForest discriminators are trained to separate real
        from synthetic data.  Mean ROC-AUC is reported directly:
            AUC = 0.50  →  indistinguishable (perfect generation)
            AUC = 1.00  →  trivially separable (terrible generation)
        Using tree-based classifiers captures non-linear distribution
        discrepancies that linear tests (e.g., logistic regression) miss.

Engineering guarantees
----------------------
- A single ColumnTransformer (built from the real data in __init__) is the
  sole encoding contract; it is reused for all sub-metrics.
- Datasets larger than *max_samples* rows are stratified-subsampled before
  KNN and C2ST computations to prevent OOM.
- KNN indexing uses the backend-agnostic NNIndex abstraction (faiss / sklearn).
- All metrics return NaN — rather than raising — when insufficient data or
  columns exist for a sub-metric, allowing partial evaluation.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as sp_stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from llm_gtd_benchmark.core.schema import DataSchema
from llm_gtd_benchmark.utils.nn_backend import NNIndex, build_nn_index
from llm_gtd_benchmark.utils.preprocessing import build_feature_encoder, stratified_subsample

logger = logging.getLogger(__name__)

_NAN = float("nan")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Dim1Result:
    """Output of :class:`FidelityEvaluator`.

    All scores follow the convention documented in the field docstrings.
    NaN is returned for any sub-metric that could not be computed (e.g., a
    dataset with no categorical columns will have NaN for TVD and Cramér's V).

    Attributes
    ----------
    ks_per_column:
        Per-column KS statistic for continuous features. Range [0, 1]; lower = better.
    tvd_per_column:
        Per-column Total Variation Distance for categorical features. Range [0, 1]; lower = better.
    mean_ks:
        Mean KS statistic across all continuous columns.
    mean_tvd:
        Mean TVD across all categorical columns.
    pearson_matrix_error:
        Mean absolute error between real and synthetic Pearson correlation matrices
        (off-diagonal elements only). Range [0, 2]; lower = better.
    cramerv_matrix_error:
        Mean absolute error between real and synthetic Cramér's V matrices
        (off-diagonal elements only). Range [0, 1]; lower = better.
    alpha_precision:
        Fraction of synthetic points falling inside the real data's k-NN support.
        Range [0, 1]; higher = better (fidelity / anti-hallucination).
    beta_recall:
        Fraction of real data's support covered by at least one synthetic point.
        Range [0, 1]; higher = better (diversity / anti-mode-collapse).
    c2st_auc_xgb:
        XGBoost discriminator ROC-AUC. 0.50 = perfect; 1.00 = trivially separable.
    c2st_auc_rf:
        RandomForest discriminator ROC-AUC.
    c2st_auc_mean:
        Mean of the two C2ST AUC scores.
    skipped_columns:
        Columns excluded from marginal distribution metrics, mapped to the
        reason for exclusion (e.g. ``"all-NaN in synthetic data"``).
        Empty dict when all columns were successfully evaluated.
    """

    # Low-order
    ks_per_column: Dict[str, float] = field(default_factory=dict)
    tvd_per_column: Dict[str, float] = field(default_factory=dict)
    mean_ks: float = _NAN
    mean_tvd: float = _NAN

    # Mid-order
    pearson_matrix_error: float = _NAN
    cramerv_matrix_error: float = _NAN

    # High-order manifold
    alpha_precision: float = _NAN
    beta_recall: float = _NAN

    # C2ST
    c2st_auc_xgb: float = _NAN
    c2st_auc_rf: float = _NAN
    c2st_auc_mean: float = _NAN

    # Diagnostics
    skipped_columns: Dict[str, str] = field(default_factory=dict)

    # Bootstrap CIs (populated only when n_boot > 0 is passed to evaluate())
    mean_ks_ci: Optional[Tuple[float, float]] = None
    mean_tvd_ci: Optional[Tuple[float, float]] = None
    alpha_precision_ci: Optional[Tuple[float, float]] = None
    beta_recall_ci: Optional[Tuple[float, float]] = None

    @property
    def summary(self) -> str:
        def _fmt(v: float) -> str:
            return f"{v:.4f}" if not np.isnan(v) else "  N/A "

        lines = [
            "── Dimension 1: Distributional Fidelity ─────────────────",
            "  Marginal distributions:",
            f"    Mean KS statistic (↓)       : {_fmt(self.mean_ks)}",
            f"    Mean TVD          (↓)       : {_fmt(self.mean_tvd)}",
            "  Pairwise correlations:",
            f"    Pearson matrix error (↓)    : {_fmt(self.pearson_matrix_error)}",
            f"    CramerV matrix error (↓)    : {_fmt(self.cramerv_matrix_error)}",
            "  High-order manifold (Alaa 2022):",
            f"    α-precision       (↑)       : {_fmt(self.alpha_precision)}",
            f"    β-recall          (↑)       : {_fmt(self.beta_recall)}",
            "  C2ST  (AUC → 0.50 = perfect):",
            f"    XGBoost AUC                 : {_fmt(self.c2st_auc_xgb)}",
            f"    RandomForest AUC            : {_fmt(self.c2st_auc_rf)}",
            f"    Mean AUC                    : {_fmt(self.c2st_auc_mean)}",
        ]
        if self.skipped_columns:
            lines.append("  Skipped columns (excluded from marginal metrics):")
            for col, reason in self.skipped_columns.items():
                lines.append(f"    {col:<30}: {reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


class FidelityEvaluator:
    """Dimension 1 distributional fidelity and diversity evaluator.

    Parameters
    ----------
    schema:
        :class:`~llm_gtd_benchmark.core.schema.DataSchema` built from the
        real training data.
    real_df:
        The reference training DataFrame.
    n_neighbors:
        *k* for k-NN manifold metrics (α-precision, β-recall).  Default: 5.
    max_samples:
        Hard cap on rows fed to manifold and C2ST computations.  Datasets
        exceeding this limit are stratified-subsampled.  Default: 50 000.
    c2st_n_splits:
        Number of folds in StratifiedKFold cross-validation for C2ST.
        Default: 5.
    random_state:
        Seed for all stochastic operations.  Default: 42.
    nn_backend:
        Nearest-neighbor backend: ``"auto"``, ``"faiss"``, or ``"sklearn"``.
        Default: ``"auto"`` (faiss when available and dataset is large).
    c2st_strat_col:
        Optional column name used to stratify the C2ST subsample.  Typically
        the target label.  Default: None (uniform subsampling).

    Examples
    --------
    >>> schema  = DataSchema(real_df)
    >>> result0 = StructuralInterceptor(schema).evaluate(synth_df)
    >>> result1 = FidelityEvaluator(schema, real_df).evaluate(result0.clean_df)
    >>> print(result1.summary)
    """

    def __init__(
        self,
        schema: DataSchema,
        real_df: pd.DataFrame,
        n_neighbors: int = 5,
        max_samples: int = 50_000,
        c2st_n_splits: int = 5,
        random_state: int = 42,
        nn_backend: str = "auto",
        c2st_strat_col: Optional[str] = None,
    ) -> None:
        self.schema = schema
        self.real_df = real_df.reset_index(drop=True)
        self.n_neighbors = n_neighbors
        self.max_samples = max_samples
        self.c2st_n_splits = c2st_n_splits
        self.random_state = random_state
        self.nn_backend = nn_backend
        self.c2st_strat_col = c2st_strat_col

        # Fit the shared encoder once on real data.
        self._encoder: ColumnTransformer = build_feature_encoder(schema, real_df)

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        clean_synth_df: pd.DataFrame,
        n_boot: int = 0,
        boot_ci: float = 0.95,
    ) -> Dim1Result:
        """Compute all Dimension 1 fidelity metrics.

        Parameters
        ----------
        clean_synth_df:
            Output of :meth:`StructuralInterceptor.evaluate` — a validated,
            downcast DataFrame guaranteed to be structurally valid.
        n_boot:
            Number of bootstrap replicates for sampling-uncertainty CIs.
            ``0`` (default) disables bootstrapping.  Recommended: 1000.
        boot_ci:
            Confidence level for bootstrap CIs.  Default: 0.95.

        Returns
        -------
        :class:`Dim1Result`
        """
        synth = clean_synth_df.reset_index(drop=True)
        real = self.real_df

        logger.info("Dim1: computing marginal distributions...")
        ks_scores, tvd_scores, skipped_cols = self._calc_column_density(real, synth)

        logger.info("Dim1: computing pairwise correlation matrices...")
        pearson_err, cramerv_err = self._calc_pairwise_correlation(real, synth)

        logger.info("Dim1: computing manifold metrics (α-precision, β-recall)...")
        alpha, beta, alpha_bool, beta_bool = self._calc_manifold_metrics(real, synth)

        logger.info("Dim1: computing C2ST (XGBoost + RandomForest)...")
        auc_xgb, auc_rf = self._calc_c2st(real, synth)

        mean_ks = float(np.nanmean(list(ks_scores.values()))) if ks_scores else _NAN
        mean_tvd = float(np.nanmean(list(tvd_scores.values()))) if tvd_scores else _NAN
        c2st_vals = [v for v in (auc_xgb, auc_rf) if not np.isnan(v)]
        c2st_mean = float(np.mean(c2st_vals)) if c2st_vals else _NAN

        result = Dim1Result(
            ks_per_column=ks_scores,
            tvd_per_column=tvd_scores,
            mean_ks=mean_ks,
            mean_tvd=mean_tvd,
            pearson_matrix_error=pearson_err,
            cramerv_matrix_error=cramerv_err,
            alpha_precision=alpha,
            beta_recall=beta,
            c2st_auc_xgb=auc_xgb,
            c2st_auc_rf=auc_rf,
            c2st_auc_mean=c2st_mean,
            skipped_columns=skipped_cols,
        )

        # ── Bootstrap CIs (opt-in) ────────────────────────────────────────────
        if n_boot > 0:
            logger.info("Dim1: bootstrapping CIs (n_boot=%d, ci=%.2f)...", n_boot, boot_ci)
            from llm_gtd_benchmark.utils.bootstrap import (
                bootstrap_mean_ci,
                bootstrap_proportion_ci,
            )

            rng = np.random.default_rng(self.random_state)

            # mean_ks CI — resample per-column KS scores
            if ks_scores:
                ks_arr = np.array(list(ks_scores.values()), dtype=np.float64)
                ks_arr = ks_arr[~np.isnan(ks_arr)]
                if len(ks_arr) > 0:
                    result.mean_ks_ci = bootstrap_mean_ci(ks_arr, n_boot, boot_ci, rng)

            # mean_tvd CI — resample per-column TVD scores
            if tvd_scores:
                tvd_arr = np.array(list(tvd_scores.values()), dtype=np.float64)
                tvd_arr = tvd_arr[~np.isnan(tvd_arr)]
                if len(tvd_arr) > 0:
                    result.mean_tvd_ci = bootstrap_mean_ci(tvd_arr, n_boot, boot_ci, rng)

            # alpha_precision CI — resample per-synth-row boolean indicators
            if alpha_bool is not None and len(alpha_bool) > 0:
                result.alpha_precision_ci = bootstrap_proportion_ci(
                    alpha_bool.astype(np.float64), n_boot, boot_ci, rng
                )

            # beta_recall CI — resample per-real-row boolean indicators
            if beta_bool is not None and len(beta_bool) > 0:
                result.beta_recall_ci = bootstrap_proportion_ci(
                    beta_bool.astype(np.float64), n_boot, boot_ci, rng
                )

        return result

    # ── Low-order: marginal distributions ────────────────────────────────────

    def _calc_column_density(
        self,
        real: pd.DataFrame,
        synth: pd.DataFrame,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, str]]:
        """KS statistic (continuous) and TVD (categorical) per column.

        Returns
        -------
        ks_scores:
            Per-column KS statistics for continuous columns.
        tvd_scores:
            Per-column TVD for categorical columns.
        skipped:
            Columns excluded from scoring, mapped to the human-readable reason.
        """
        ks_scores: Dict[str, float] = {}
        tvd_scores: Dict[str, float] = {}
        skipped: Dict[str, str] = {}

        for col_schema in self.schema.columns:
            col = col_schema.name
            r_col = real[col].dropna()
            s_col = synth[col].dropna()

            if len(r_col) == 0 and len(s_col) == 0:
                reason = "all-NaN in both real and synthetic data"
                logger.warning("Column '%s' is all-NaN in real and synth; skipping.", col)
                skipped[col] = reason
                continue
            if len(r_col) == 0:
                reason = "all-NaN in real data"
                logger.warning("Column '%s' is all-NaN in real data; skipping.", col)
                skipped[col] = reason
                continue
            if len(s_col) == 0:
                reason = "all-NaN in synthetic data"
                logger.warning("Column '%s' is all-NaN in synthetic data; skipping.", col)
                skipped[col] = reason
                continue

            if col_schema.col_type == "continuous":
                stat, _ = sp_stats.ks_2samp(
                    r_col.astype(float).values,
                    s_col.astype(float).values,
                )
                ks_scores[col] = float(stat)
            else:
                tvd_scores[col] = _tvd(r_col, s_col)

        return ks_scores, tvd_scores, skipped

    # ── Mid-order: pairwise correlation matrices ──────────────────────────────

    def _calc_pairwise_correlation(
        self,
        real: pd.DataFrame,
        synth: pd.DataFrame,
    ) -> Tuple[float, float]:
        """MAE between real/synth Pearson (cont) and Cramér's V (cat) matrices."""
        cont_names = [c.name for c in self.schema.continuous_columns]
        cat_names = [c.name for c in self.schema.categorical_columns]

        pearson_err = _NAN
        if len(cont_names) >= 2:
            r_mat = real[cont_names].astype(float).corr(method="pearson").values
            s_mat = synth[cont_names].astype(float).corr(method="pearson").values
            mask = ~np.eye(len(cont_names), dtype=bool)
            pearson_err = float(np.mean(np.abs(r_mat[mask] - s_mat[mask])))

        cramerv_err = _NAN
        if len(cat_names) >= 2:
            r_cv = _cramerv_matrix(real[cat_names])
            s_cv = _cramerv_matrix(synth[cat_names])
            mask = ~np.eye(len(cat_names), dtype=bool)
            cramerv_err = float(np.mean(np.abs(r_cv[mask] - s_cv[mask])))

        return pearson_err, cramerv_err

    # ── High-order: manifold metrics ──────────────────────────────────────────

    def _calc_manifold_metrics(
        self,
        real: pd.DataFrame,
        synth: pd.DataFrame,
    ) -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray]]:
        """α-precision and β-recall via adaptive k-NN radii (Alaa et al. 2022).

        Algorithm
        ---------
        1. Encode both datasets with the shared ColumnTransformer.
        2. For each real point r, compute δ_k(r) = distance to its k-th
           nearest real neighbor (its "support radius").
        3. α-precision: for each synthetic point s, find its nearest real
           point r*; s is "authentic" iff d(s, r*) ≤ δ_k(r*).
        4. β-recall: for each real point r, find its nearest synthetic point
           s*; r is "covered" iff d(r, s*) ≤ δ_k(r).

        Returns
        -------
        Tuple of (alpha_precision, beta_recall, alpha_bool, beta_bool).
        ``alpha_bool`` and ``beta_bool`` are per-sample boolean arrays used
        for bootstrap CI computation; both are ``None`` on failure.
        """
        real_s = stratified_subsample(real, self.max_samples, random_state=self.random_state)
        synth_s = stratified_subsample(synth, self.max_samples, random_state=self.random_state)

        try:
            R = self._encoder.transform(real_s).astype(np.float32)
            S = self._encoder.transform(synth_s).astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.error("Feature encoding failed in manifold metrics: %s", exc)
            return _NAN, _NAN, None, None

        n_real = R.shape[0]
        k = min(self.n_neighbors, n_real - 1)  # guard for tiny datasets

        if k < 1:
            logger.warning("Too few real samples (%d) for k-NN (k=%d); skipping manifold.", n_real, self.n_neighbors)
            return _NAN, _NAN, None, None

        # ── Step 1: k-NN self-distances on real data (compute radii) ──────────
        # Query k+1 neighbours; index 0 is the point itself (distance ≈ 0).
        real_index: NNIndex = build_nn_index(R, force_backend=self.nn_backend)
        r_dists, _ = real_index.query(R, k=k + 1)
        radii_real = r_dists[:, k].astype(np.float64)  # k-th true neighbour distance

        # ── Step 2: α-precision ───────────────────────────────────────────────
        s_to_r_dists, s_to_r_idx = real_index.query(S, k=1)
        s_to_r_dists = s_to_r_dists[:, 0].astype(np.float64)
        nearest_radii = radii_real[s_to_r_idx[:, 0]]
        alpha_bool = (s_to_r_dists <= nearest_radii)   # shape (n_synth,)
        alpha_precision = float(np.mean(alpha_bool))

        # ── Step 3: β-recall ──────────────────────────────────────────────────
        synth_index: NNIndex = build_nn_index(S, force_backend=self.nn_backend)
        r_to_s_dists, _ = synth_index.query(R, k=1)
        r_to_s_dists = r_to_s_dists[:, 0].astype(np.float64)
        beta_bool = (r_to_s_dists <= radii_real)       # shape (n_real,)
        beta_recall = float(np.mean(beta_bool))

        logger.info("Manifold: α-precision=%.4f, β-recall=%.4f.", alpha_precision, beta_recall)
        return alpha_precision, beta_recall, alpha_bool, beta_bool

    # ── Global: classifier two-sample test ───────────────────────────────────

    def _calc_c2st(
        self,
        real: pd.DataFrame,
        synth: pd.DataFrame,
    ) -> Tuple[float, float]:
        """Train XGBoost and RandomForest discriminators; return mean ROC-AUC.

        Real data is labelled 1; synthetic data is labelled 0.
        AUC = 0.50 → indistinguishable (ideal).
        AUC = 1.00 → trivially separable (worst case).
        """
        real_s = stratified_subsample(
            real, self.max_samples, strat_col=self.c2st_strat_col, random_state=self.random_state
        )
        synth_s = stratified_subsample(
            synth, self.max_samples, strat_col=self.c2st_strat_col, random_state=self.random_state
        )

        try:
            X_real = self._encoder.transform(real_s)
            X_synth = self._encoder.transform(synth_s)
        except Exception as exc:  # noqa: BLE001
            logger.error("Feature encoding failed in C2ST: %s", exc)
            return _NAN, _NAN

        X = np.vstack([X_real, X_synth])
        y = np.concatenate([np.ones(len(X_real)), np.zeros(len(X_synth))])

        cv = StratifiedKFold(
            n_splits=self.c2st_n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        # ── XGBoost ───────────────────────────────────────────────────────────
        auc_xgb = _NAN
        try:
            from xgboost import XGBClassifier

            xgb = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
            )
            auc_xgb = float(np.mean(cross_val_score(xgb, X, y, cv=cv, scoring="roc_auc")))
            logger.info("C2ST XGBoost AUC: %.4f", auc_xgb)
        except ImportError:
            warnings.warn(
                "xgboost is not installed; C2ST XGBoost AUC will be NaN. "
                "Install it with:  pip install xgboost",
                ImportWarning,
                stacklevel=3,
            )

        # ── RandomForest ──────────────────────────────────────────────────────
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            n_jobs=-1,
            random_state=self.random_state,
        )
        auc_rf = float(np.mean(cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")))
        logger.info("C2ST RandomForest AUC: %.4f", auc_rf)

        return auc_xgb, auc_rf


# ---------------------------------------------------------------------------
# Module-level statistical helpers
# ---------------------------------------------------------------------------


def _tvd(real_series: pd.Series, synth_series: pd.Series) -> float:
    """Total Variation Distance: 0.5 · Σ|p_i − q_i|.

    Operates on empirical frequencies; categories absent from one distribution
    contribute their full probability mass to the sum.
    """
    all_cats = set(real_series.unique()) | set(synth_series.unique())
    r_freq = real_series.value_counts(normalize=True)
    s_freq = synth_series.value_counts(normalize=True)
    tvd = 0.5 * sum(
        abs(r_freq.get(c, 0.0) - s_freq.get(c, 0.0)) for c in all_cats
    )
    return float(tvd)


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V (Bergsma & Wicher, 2013).

    Returns 0.0 when the contingency table is degenerate (single row or column).
    """
    from scipy.stats import chi2_contingency

    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0

    chi2, _, _, _ = chi2_contingency(ct)
    n = float(ct.values.sum())
    r, k = ct.shape

    phi2 = chi2 / n
    # Bias correction
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def _cramerv_matrix(df: pd.DataFrame) -> np.ndarray:
    """Compute the full pairwise bias-corrected Cramér's V matrix for *df*."""
    cols = df.columns.tolist()
    n = len(cols)
    mat = np.eye(n)  # diagonal = 1 by definition
    for i in range(n):
        for j in range(i + 1, n):
            v = _cramers_v(df[cols[i]], df[cols[j]])
            mat[i, j] = mat[j, i] = v
    return mat

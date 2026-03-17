"""
Dataset Auto-Profiler — column-level and dataset-level statistical summary.

Design principles
-----------------
- **No composite score.**  Composite scores introduce arbitrary weighting;
  callers decide which dimensions matter for their use-case.
- **No new heavy dependencies.**  Everything runs on pandas + scipy + numpy
  (already required by the benchmark).
- Functional dependencies use :func:`~llm_gtd_benchmark.core.logic_spec.discover_fds`
  so the FD discovery algorithm stays in one canonical place.
- Bimodality is detected via the Bimodality Coefficient (BC, Pfister et al. 2013):
      BC = (γ₁² + 1) / (γ₂ + 3·(n-1)²/((n-2)·(n-3)))
  where γ₁ = skewness, γ₂ = excess kurtosis.  BC > 5/9 ≈ 0.556 signals
  non-unimodality without requiring kernel density estimation.
- Cramér's V pairs measure categorical inter-column associations.
- Pearson pairs measure continuous inter-column associations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

logger = logging.getLogger(__name__)

_NAN = float("nan")
_BC_THRESHOLD = 5.0 / 9.0  # ≈ 0.556 — Pfister et al. 2013


# ---------------------------------------------------------------------------
# Column-level profile
# ---------------------------------------------------------------------------


@dataclass
class ColumnProfile:
    """Statistical profile of a single DataFrame column.

    Attributes
    ----------
    name:
        Column name.
    col_type:
        ``"continuous"`` or ``"categorical"`` (inferred from the data).
    missing_rate:
        Fraction of values that are ``NaN`` or ``None``.  Range [0, 1].
    n_valid:
        Number of non-null values.

    Continuous-only attributes (``NaN`` for categorical columns)
    -------------------------------------------------------------
    mean, std, min, max:
        Descriptive statistics on non-null values.
    skewness:
        Fisher's skewness (third standardised moment).  0 = symmetric.
    kurtosis:
        Excess kurtosis (fourth standardised moment − 3).  0 = normal.
    is_multimodal:
        ``True`` when BC > 5/9 (Pfister et al. 2013), indicating the
        marginal distribution is likely non-unimodal.

    Categorical-only attributes (``NaN`` / ``None`` for continuous columns)
    -----------------------------------------------------------------------
    n_unique:
        Number of distinct non-null values.
    top_value:
        Most frequent category (``None`` when all values are null).
    top_freq_ratio:
        Relative frequency of ``top_value``.  Range [0, 1].
    """

    name: str
    col_type: str  # "continuous" | "categorical"
    missing_rate: float
    n_valid: int

    # Continuous
    mean: float = _NAN
    std: float = _NAN
    min: float = _NAN
    max: float = _NAN
    skewness: float = _NAN
    kurtosis: float = _NAN
    is_multimodal: bool = False

    # Categorical
    n_unique: Optional[int] = None
    top_value: Optional[str] = None
    top_freq_ratio: float = _NAN


# ---------------------------------------------------------------------------
# Dataset-level profile
# ---------------------------------------------------------------------------


@dataclass
class DatasetProfile:
    """Aggregated statistical profile of an entire DataFrame.

    Attributes
    ----------
    dataset_name:
        Human-readable label (passed to :meth:`DatasetProfiler.profile`).
    n_rows:
        Total number of rows.
    n_cols:
        Total number of columns.
    total_missing_rate:
        Overall fraction of missing values across the entire DataFrame.
    column_profiles:
        Per-column :class:`ColumnProfile` objects, keyed by column name.
    top_pearson_pairs:
        Up to ``top_k`` highest-|Pearson| pairs among continuous columns.
        Each entry is ``(col_a, col_b, pearson_r)``.
    top_cramer_pairs:
        Up to ``top_k`` highest-Cramér's-V pairs among categorical columns.
        Each entry is ``(col_a, col_b, cramer_v)``.
    fd_candidates:
        Approximate functional dependencies found by
        :func:`~llm_gtd_benchmark.core.logic_spec.discover_fds`.
        Each entry is ``(determinant_col, dependent_col, violation_rate)``.
    class_imbalance_ratio:
        For datasets inferred to have a binary or multi-class target, the
        ratio of the most-frequent class to the least-frequent class.
        ``NaN`` when no target column is available or the dataset has no
        obvious categorical column with 2–20 unique values.
    """

    dataset_name: str
    n_rows: int
    n_cols: int
    total_missing_rate: float
    column_profiles: Dict[str, ColumnProfile] = field(default_factory=dict)
    top_pearson_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    top_cramer_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    fd_candidates: List[Tuple[str, str, float]] = field(default_factory=list)
    class_imbalance_ratio: float = _NAN

    @property
    def summary(self) -> str:
        """Human-readable summary string."""
        cont_profiles = [p for p in self.column_profiles.values() if p.col_type == "continuous"]
        cat_profiles = [p for p in self.column_profiles.values() if p.col_type == "categorical"]
        multimodal = [p.name for p in cont_profiles if p.is_multimodal]
        high_missing = [p.name for p in self.column_profiles.values() if p.missing_rate > 0.1]

        lines = [
            f"── Dataset Profile: {self.dataset_name} ───────────────────────────",
            f"  Rows: {self.n_rows:,}   Cols: {self.n_cols}   "
            f"Overall missing: {self.total_missing_rate:.2%}",
            f"  Continuous: {len(cont_profiles)}   Categorical: {len(cat_profiles)}",
        ]
        if multimodal:
            lines.append(f"  Possibly multimodal (BC > 5/9): {', '.join(multimodal)}")
        if high_missing:
            lines.append(f"  High missing (> 10 %): {', '.join(high_missing)}")
        if not np.isnan(self.class_imbalance_ratio):
            lines.append(f"  Class imbalance ratio (max/min freq): {self.class_imbalance_ratio:.2f}")
        if self.top_pearson_pairs:
            lines.append("  Top Pearson pairs:")
            for a, b, r in self.top_pearson_pairs:
                lines.append(f"    {a} × {b}: r = {r:+.3f}")
        if self.top_cramer_pairs:
            lines.append("  Top Cramér's V pairs:")
            for a, b, v in self.top_cramer_pairs:
                lines.append(f"    {a} × {b}: V = {v:.3f}")
        if self.fd_candidates:
            lines.append("  Approx. functional dependencies (violation rate < 5 %):")
            for det, dep, vr in self.fd_candidates:
                lines.append(f"    {det} → {dep}  (violation rate: {vr:.2%})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bimodality coefficient helper
# ---------------------------------------------------------------------------


def _bimodality_coefficient(skewness: float, excess_kurtosis: float, n: int) -> float:
    """Bimodality Coefficient (Pfister et al. 2013).

    BC = (γ₁² + 1) / (γ₂ + 3·(n-1)²/((n-2)·(n-3)))

    where γ₁ is skewness and γ₂ is excess kurtosis.

    Returns NaN when n < 4 (denominator involves n-2, n-3).
    BC > 5/9 ≈ 0.556 indicates likely non-unimodality.
    """
    if n < 4 or np.isnan(skewness) or np.isnan(excess_kurtosis):
        return _NAN
    numerator = skewness ** 2 + 1.0
    # Finite-sample correction term for denominator
    denom_correction = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    denominator = excess_kurtosis + denom_correction
    if denominator == 0.0:
        return _NAN
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# Cramér's V (bias-corrected)
# ---------------------------------------------------------------------------


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V.  Returns 0.0 on degenerate tables."""
    from scipy.stats import chi2_contingency

    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2, _, _, _ = chi2_contingency(ct)
    n = float(ct.values.sum())
    r, k = ct.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


# ---------------------------------------------------------------------------
# DatasetProfiler
# ---------------------------------------------------------------------------


class DatasetProfiler:
    """Compute a statistical profile of a DataFrame.

    Parameters
    ----------
    top_k_pairs:
        Number of top association pairs to report for Pearson and Cramér's V.
        Default: 5.
    fd_max_violation_rate:
        Maximum violation rate for a column pair to be included as an FD
        candidate.  Default: 0.05 (5 %).
    fd_min_support:
        Minimum number of rows a determinant value must appear in for its
        violation count to be meaningful.  Passed through to
        :func:`~llm_gtd_benchmark.core.logic_spec.discover_fds`.
        Default: 10.
    infer_target_col:
        When ``True``, automatically identify a binary/multi-class target
        column (categorical with 2–20 unique values, lowest overall missing
        rate wins) and compute a class imbalance ratio.  Default: ``True``.

    Examples
    --------
    >>> profiler = DatasetProfiler()
    >>> profile = profiler.profile(df, dataset_name="adult")
    >>> print(profile.summary)
    >>> profile.column_profiles["age"].is_multimodal
    """

    def __init__(
        self,
        top_k_pairs: int = 5,
        fd_max_violation_rate: float = 0.05,
        fd_min_support: int = 10,
        infer_target_col: bool = True,
    ) -> None:
        self.top_k_pairs = top_k_pairs
        self.fd_max_violation_rate = fd_max_violation_rate
        self.fd_min_support = fd_min_support
        self.infer_target_col = infer_target_col

    # ── Public API ────────────────────────────────────────────────────────────

    def profile(self, df: pd.DataFrame, dataset_name: str = "dataset") -> DatasetProfile:
        """Compute a full statistical profile of *df*.

        Parameters
        ----------
        df:
            The DataFrame to profile.  Any dtype mix is acceptable.
        dataset_name:
            Human-readable label stored in the returned :class:`DatasetProfile`.

        Returns
        -------
        :class:`DatasetProfile`
        """
        n_rows, n_cols = df.shape
        total_missing = float(df.isna().values.sum()) / max(1, df.size)

        # ── Per-column profiles ───────────────────────────────────────────────
        col_profiles: Dict[str, ColumnProfile] = {}
        cont_cols: List[str] = []
        cat_cols: List[str] = []

        for col in df.columns:
            cp = self._profile_column(df[col])
            col_profiles[col] = cp
            if cp.col_type == "continuous":
                cont_cols.append(col)
            else:
                cat_cols.append(col)

        # ── Top Pearson pairs ─────────────────────────────────────────────────
        top_pearson: List[Tuple[str, str, float]] = []
        if len(cont_cols) >= 2:
            top_pearson = self._top_pearson_pairs(df[cont_cols], self.top_k_pairs)

        # ── Top Cramér's V pairs ──────────────────────────────────────────────
        top_cramer: List[Tuple[str, str, float]] = []
        if len(cat_cols) >= 2:
            top_cramer = self._top_cramer_pairs(df[cat_cols], self.top_k_pairs)

        # ── Functional dependencies ───────────────────────────────────────────
        # discover_fds returns strict FDs (violation_rate == 0 by construction).
        # We additionally compute violation rates to respect fd_max_violation_rate.
        fd_candidates: List[Tuple[str, str, float]] = []
        try:
            from llm_gtd_benchmark.core.schema import DataSchema as _DS
            from llm_gtd_benchmark.core.logic_spec import discover_fds

            # discover_fds requires a DataSchema; build a lightweight one if needed
            _tmp_schema = _DS(df)
            fds = discover_fds(df, _tmp_schema, min_rows=self.fd_min_support)
            # fds is List[Tuple[str, str]] — each entry is (determinant, dependent)
            for det, dep in fds:
                vr = self._fd_violation_rate(df, det, dep, self.fd_min_support)
                # Strict FDs found by discover_fds have vr == 0; include them all
                if not np.isnan(vr) and vr <= self.fd_max_violation_rate:
                    fd_candidates.append((det, dep, vr))
        except Exception as exc:
            logger.warning("DatasetProfiler: FD discovery failed — %s", exc)

        # ── Class imbalance ───────────────────────────────────────────────────
        class_imbalance: float = _NAN
        if self.infer_target_col and cat_cols:
            class_imbalance = self._infer_class_imbalance(df, cat_cols)

        return DatasetProfile(
            dataset_name=dataset_name,
            n_rows=n_rows,
            n_cols=n_cols,
            total_missing_rate=total_missing,
            column_profiles=col_profiles,
            top_pearson_pairs=top_pearson,
            top_cramer_pairs=top_cramer,
            fd_candidates=fd_candidates,
            class_imbalance_ratio=class_imbalance,
        )

    # ── Column profiling ──────────────────────────────────────────────────────

    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        """Build a :class:`ColumnProfile` for a single Series."""
        n_total = len(series)
        n_null = int(series.isna().sum())
        n_valid = n_total - n_null
        missing_rate = n_null / max(1, n_total)

        # Determine col_type: try numeric coercion
        numeric = pd.to_numeric(series.dropna(), errors="coerce")
        is_continuous = (numeric.notna().sum() / max(1, len(numeric))) >= 0.9

        if is_continuous and n_valid > 0:
            vals = numeric.dropna().values.astype(float)
            n_vals = len(vals)
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if n_vals > 1 else _NAN
            min_v = float(np.min(vals))
            max_v = float(np.max(vals))
            skew_v = float(skew(vals)) if n_vals >= 3 else _NAN
            kurt_v = float(kurtosis(vals, fisher=True)) if n_vals >= 4 else _NAN  # excess kurtosis
            bc = _bimodality_coefficient(skew_v, kurt_v, n_vals)
            is_mm = (not np.isnan(bc)) and (bc > _BC_THRESHOLD)

            return ColumnProfile(
                name=series.name,
                col_type="continuous",
                missing_rate=missing_rate,
                n_valid=n_valid,
                mean=mean_v,
                std=std_v,
                min=min_v,
                max=max_v,
                skewness=skew_v,
                kurtosis=kurt_v,
                is_multimodal=is_mm,
            )
        else:
            valid_vals = series.dropna()
            n_unique = int(valid_vals.nunique())
            top_value: Optional[str] = None
            top_freq_ratio: float = _NAN

            if n_unique > 0:
                vc = valid_vals.value_counts(normalize=True)
                top_value = str(vc.index[0])
                top_freq_ratio = float(vc.iloc[0])

            return ColumnProfile(
                name=series.name,
                col_type="categorical",
                missing_rate=missing_rate,
                n_valid=n_valid,
                n_unique=n_unique,
                top_value=top_value,
                top_freq_ratio=top_freq_ratio,
            )

    # ── Association pairs ─────────────────────────────────────────────────────

    def _top_pearson_pairs(
        self, cont_df: pd.DataFrame, top_k: int
    ) -> List[Tuple[str, str, float]]:
        """Compute pairwise Pearson correlations; return top-|r| pairs."""
        cols = cont_df.columns.tolist()
        try:
            corr = cont_df.astype(float).corr(method="pearson").values
        except Exception as exc:
            logger.warning("DatasetProfiler: Pearson matrix failed — %s", exc)
            return []

        pairs: List[Tuple[str, str, float]] = []
        n = len(cols)
        for i in range(n):
            for j in range(i + 1, n):
                r = corr[i, j]
                if not np.isnan(r):
                    pairs.append((cols[i], cols[j], float(r)))

        pairs.sort(key=lambda t: abs(t[2]), reverse=True)
        return pairs[:top_k]

    def _top_cramer_pairs(
        self, cat_df: pd.DataFrame, top_k: int
    ) -> List[Tuple[str, str, float]]:
        """Compute pairwise bias-corrected Cramér's V; return top-V pairs."""
        cols = cat_df.columns.tolist()
        pairs: List[Tuple[str, str, float]] = []
        n = len(cols)

        for i in range(n):
            for j in range(i + 1, n):
                try:
                    v = _cramers_v(cat_df[cols[i]], cat_df[cols[j]])
                    pairs.append((cols[i], cols[j], v))
                except Exception as exc:
                    logger.debug(
                        "DatasetProfiler: CramerV failed for (%s, %s) — %s",
                        cols[i], cols[j], exc,
                    )

        pairs.sort(key=lambda t: t[2], reverse=True)
        return pairs[:top_k]

    # ── Functional dependency violation rate ──────────────────────────────────

    @staticmethod
    def _fd_violation_rate(
        df: pd.DataFrame, det: str, dep: str, min_support: int
    ) -> float:
        """Compute the violation rate of the FD ``det → dep``.

        A violation occurs when the same determinant value maps to multiple
        distinct dependent values.  Only groups with >= min_support rows are
        checked (small groups are excluded to reduce false positives).

        Returns NaN when insufficient data exists.
        """
        if det not in df.columns or dep not in df.columns:
            return _NAN
        sub = df[[det, dep]].dropna()
        if len(sub) == 0:
            return _NAN

        grouped = sub.groupby(det)[dep].nunique()
        large_groups_mask = sub.groupby(det)[dep].count() >= min_support
        relevant = grouped[large_groups_mask]

        if len(relevant) == 0:
            return _NAN

        n_violating = int((relevant > 1).sum())
        total_rows = int(sub[sub[det].isin(relevant.index)].shape[0])
        if total_rows == 0:
            return _NAN

        # Fraction of rows in groups that violate the FD
        violated_rows = int(
            sub[sub[det].isin(relevant[relevant > 1].index)].shape[0]
        )
        return float(violated_rows / total_rows)

    # ── Class imbalance ───────────────────────────────────────────────────────

    @staticmethod
    def _infer_class_imbalance(df: pd.DataFrame, cat_cols: List[str]) -> float:
        """Identify the most-likely target column and return its imbalance ratio.

        Heuristic: select the categorical column with 2–20 unique values that
        has the **lowest missing rate** (least missing → most likely a target
        label).

        Returns NaN if no eligible column is found.
        """
        candidates = []
        for col in cat_cols:
            valid = df[col].dropna()
            n_unique = valid.nunique()
            if 2 <= n_unique <= 20:
                missing_rate = float(df[col].isna().mean())
                candidates.append((col, missing_rate, n_unique))

        if not candidates:
            return _NAN

        # Pick the column with the lowest missing rate
        target_col = min(candidates, key=lambda t: t[1])[0]
        vc = df[target_col].dropna().value_counts()

        if len(vc) < 2:
            return _NAN

        max_freq = float(vc.max())
        min_freq = float(vc.min())
        if min_freq == 0:
            return _NAN

        return float(max_freq / min_freq)

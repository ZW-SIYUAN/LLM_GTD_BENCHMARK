"""
Statistical Significance Testing for multi-model benchmark comparisons.

Two evaluation regimes are supported:

Multi-run (≥ 2 matched runs per model)
---------------------------------------
Given *k* matched runs of model A and model B evaluated on the same dataset,
for each metric:
- k ≥ 5: Wilcoxon signed-rank test (non-parametric, no normality assumption).
- 2 ≤ k ≤ 4: Paired t-test (sensitive to non-normality at small n; prefer
  Wilcoxon when possible).

Single-run (1 run per model, requires bootstrap CIs)
-----------------------------------------------------
When each model was evaluated exactly once *but* bootstrap CIs were computed
during evaluation (``n_boot > 0`` passed to ``evaluate()``), non-overlapping
CIs are used as a significance proxy.  This approach is conservative — CI
overlap does not imply significance, but non-overlap strongly suggests it.

Multiple testing correction
---------------------------
All raw p-values are Holm–Bonferroni step-down corrected (Holm 1979) across
the *m* metrics tested in a single comparison.  Holm–Bonferroni is uniformly
more powerful than Bonferroni while still controlling the family-wise error
rate at the nominal α level.

Metric registry
---------------
Known benchmark metrics are pre-registered with:
- The attribute path to extract the scalar from a ResultBundle entry
  (e.g. ``"result1.mean_ks"``).
- Whether higher values are better (``higher_is_better``).
- An optional CI path for the single-run regime (e.g. ``"result1.mean_ks_ci"``).

Callers may pass ``metrics=None`` to test all registered metrics, or supply
a custom list of metric names to restrict the test.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_NAN = float("nan")

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

# Format: metric_name → (attribute_path, higher_is_better, ci_path_or_None)
# attribute_path uses dot notation relative to a ResultBundle object.
_METRIC_REGISTRY: Dict[str, Tuple[str, bool, Optional[str]]] = {
    # Dim 0 – structural validity
    "irr": ("result0.irr", True, None),
    # Dim 1 – fidelity
    "mean_ks": ("result1.mean_ks", False, "result1.mean_ks_ci"),
    "mean_tvd": ("result1.mean_tvd", False, "result1.mean_tvd_ci"),
    "alpha_precision": ("result1.alpha_precision", True, "result1.alpha_precision_ci"),
    "beta_recall": ("result1.beta_recall", True, "result1.beta_recall_ci"),
    "c2st_auc_mean": ("result1.c2st_auc_mean", False, None),
    "pearson_matrix_error": ("result1.pearson_matrix_error", False, None),
    "cramerv_matrix_error": ("result1.cramerv_matrix_error", False, None),
    # Dim 2 – logical consistency
    "icvr": ("result2.icvr", True, None),
    "hcs_violation_rate": ("result2.hcs_violation_rate", False, None),
    "mdi_mean": ("result2.mdi_mean", True, None),
    "dsi_gap": ("result2.dsi_gap", False, None),
    # Dim 3 – ML utility (primary: macro F1 / RMSE; see note below)
    # Dim 3 stores dicts; use the first available key at extraction time.
    # Registered as special entries; extraction handles dict averaging.
    "mle_tstr_primary": ("result3.mle_tstr", True, None),   # extracted as mean of dict
    "lle_tstr_primary": ("result3.lle_tstr", True, None),
    # Dim 4 – privacy
    "dcr_5th_percentile": ("result4.dcr_5th_percentile", True, "result4.dcr_5th_ci"),
    "exact_match_rate": ("result4.exact_match_rate", False, "result4.exact_match_rate_ci"),
    # Dim 5 – fairness
    "delta_eo_mean": ("result5.delta_eo", False, "result5.delta_eo_ci"),     # mean over cols
    "delta_dp_mean": ("result5.delta_dp", False, "result5.delta_dp_ci"),
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MetricTestResult:
    """Statistical test result for a single metric pair.

    Attributes
    ----------
    metric:
        Registered metric name (key in ``_METRIC_REGISTRY``).
    value_a:
        Scalar metric value for model A (mean over runs if multi-run).
    value_b:
        Scalar metric value for model B.
    difference:
        ``value_b − value_a``.  Positive means B outperforms A when
        ``higher_is_better=True``; negative when ``higher_is_better=False``.
    test_method:
        One of: ``"wilcoxon"``, ``"paired_t"``, ``"ci_overlap"``, ``"na"``.
    p_value:
        Raw p-value from the test.  ``NaN`` for ``"ci_overlap"`` regime.
    adjusted_p_value:
        Holm–Bonferroni corrected p-value.  ``NaN`` for single-run regime.
    significant:
        Whether the difference is statistically significant at the given α.
        For CI-overlap: ``True`` iff the CIs do not overlap.
    effect_size:
        Rank-biserial correlation (Wilcoxon) or Cohen's d (paired t).
        ``NaN`` for CI-overlap or when not computable.
    ci_overlap:
        ``True`` iff the bootstrap CIs overlap (single-run regime only).
        ``None`` when CIs are not available.
    """

    metric: str
    value_a: float
    value_b: float
    difference: float
    test_method: str
    p_value: float = _NAN
    adjusted_p_value: float = _NAN
    significant: bool = False
    effect_size: float = _NAN
    ci_overlap: Optional[bool] = None


@dataclass
class SignificanceReport:
    """Report from :class:`SignificanceTester.compare`.

    Attributes
    ----------
    model_a:
        Name of model A.
    model_b:
        Name of model B.
    n_runs_a:
        Number of evaluation runs for model A.
    n_runs_b:
        Number of evaluation runs for model B.
    alpha:
        Family-wise error rate threshold used for significance calls.
    correction:
        Multiple-testing correction method (currently only ``"holm"``).
    results:
        Per-metric :class:`MetricTestResult` objects, keyed by metric name.
    """

    model_a: str
    model_b: str
    n_runs_a: int
    n_runs_b: int
    alpha: float
    correction: str
    results: Dict[str, MetricTestResult] = field(default_factory=dict)

    @property
    def significant_metrics(self) -> List[str]:
        """Metrics where the difference is statistically significant."""
        return [k for k, v in self.results.items() if v.significant]

    @property
    def summary(self) -> str:
        """Human-readable comparison table."""
        lines = [
            f"── Significance Report: {self.model_a} vs. {self.model_b} ─────────",
            f"  Runs: A={self.n_runs_a}, B={self.n_runs_b}  |  "
            f"α={self.alpha}  |  correction={self.correction}",
            "",
            f"  {'Metric':<28} {'A':<10} {'B':<10} {'Diff':>8}  {'Method':<14} "
            f"{'adj-p':>8}  {'Sig?':<5} {'Effect':>8}",
            "  " + "─" * 100,
        ]

        def _fv(v: float) -> str:
            return f"{v:.4f}" if not np.isnan(v) else "  N/A  "

        for metric, r in self.results.items():
            sig_marker = "✓" if r.significant else ""
            adj_p_str = f"{r.adjusted_p_value:.3f}" if not np.isnan(r.adjusted_p_value) else "  N/A "
            eff_str = f"{r.effect_size:.3f}" if not np.isnan(r.effect_size) else "  N/A "
            lines.append(
                f"  {metric:<28} {_fv(r.value_a):<10} {_fv(r.value_b):<10} "
                f"{r.difference:>+8.4f}  {r.test_method:<14} "
                f"{adj_p_str:>8}  {sig_marker:<5} {eff_str:>8}"
            )

        n_sig = len(self.significant_metrics)
        lines.append(f"\n  Significant metrics ({n_sig}): {', '.join(self.significant_metrics) or 'none'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Holm–Bonferroni correction
# ---------------------------------------------------------------------------


def _holm_bonferroni(p_values: List[float], alpha: float) -> List[float]:
    """Holm–Bonferroni step-down correction.

    Parameters
    ----------
    p_values:
        Raw p-values in *original* metric order (NaN values are excluded
        from the procedure and get ``NaN`` adjusted p-values).
    alpha:
        Family-wise error rate threshold.

    Returns
    -------
    List of adjusted p-values in the same order as the input.
    """
    n = len(p_values)
    adjusted = [_NAN] * n

    # Separate valid indices (non-NaN p-values)
    valid_idx = [i for i, p in enumerate(p_values) if not np.isnan(p)]
    if not valid_idx:
        return adjusted

    m = len(valid_idx)
    # Sort valid indices by ascending p-value
    sorted_valid = sorted(valid_idx, key=lambda i: p_values[i])

    running_min = 1.0
    # Compute adjusted p-values in ascending order; enforce monotonicity
    # by carrying forward the minimum from the right (step-down procedure).
    raw_adj = []
    for rank, idx in enumerate(sorted_valid):
        # Holm: adjusted p_k = p_k × (m − k + 1),  k = 1, 2, …, m
        adj = p_values[idx] * (m - rank)
        raw_adj.append(min(adj, 1.0))

    # Enforce non-decreasing monotonicity from left to right.
    for j in range(len(raw_adj)):
        if j > 0:
            raw_adj[j] = max(raw_adj[j], raw_adj[j - 1])

    for rank, idx in enumerate(sorted_valid):
        adjusted[idx] = raw_adj[rank]

    return adjusted


# ---------------------------------------------------------------------------
# SignificanceTester
# ---------------------------------------------------------------------------


class SignificanceTester:
    """Compare two models across benchmark metrics.

    Parameters
    ----------
    alpha:
        Family-wise error rate threshold for significance calls.
        Default: 0.05.

    Examples
    --------
    Multi-run comparison (each model run 5 times):

    >>> from llm_gtd_benchmark import BenchmarkPipeline, PipelineConfig
    >>> bundles_a = [pipeline_a.run(synth_a_i) for synth_a_i in synth_runs_a]
    >>> bundles_b = [pipeline_b.run(synth_b_i) for synth_b_i in synth_runs_b]
    >>> tester = SignificanceTester(alpha=0.05)
    >>> report = tester.compare(bundles_a, bundles_b, model_a="GReaT", model_b="REaLTabFormer")
    >>> print(report.summary)

    Single-run comparison (requires ``n_boot > 0`` at evaluation time):

    >>> bundle_a = pipeline_a.run(synth_a)   # pipeline had n_boot=1000
    >>> bundle_b = pipeline_b.run(synth_b)
    >>> report = tester.compare([bundle_a], [bundle_b], model_a="GReaT", model_b="REaLTabFormer")
    >>> print(report.summary)
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.alpha = alpha

    # ── Public API ────────────────────────────────────────────────────────────

    def compare(
        self,
        bundles_a: Sequence[Any],
        bundles_b: Sequence[Any],
        model_a: str = "model_a",
        model_b: str = "model_b",
        metrics: Optional[List[str]] = None,
        correction: str = "holm",
    ) -> SignificanceReport:
        """Compare model A and model B across benchmark metrics.

        Parameters
        ----------
        bundles_a:
            :class:`~llm_gtd_benchmark.core.result_bundle.ResultBundle` objects
            for model A — one per evaluation run.
        bundles_b:
            Result bundles for model B.  Must contain the same number of runs
            as ``bundles_a`` for the paired-test regime (multi-run path).
            Single-run comparison is supported when both lists have length 1.
        model_a:
            Display name for model A.
        model_b:
            Display name for model B.
        metrics:
            Subset of metric names from ``_METRIC_REGISTRY`` to test.
            Pass ``None`` to test all registered metrics for which both
            models have non-NaN values.
        correction:
            Multiple-testing correction method.  Currently only ``"holm"``
            is supported.

        Returns
        -------
        :class:`SignificanceReport`
        """
        if not bundles_a or not bundles_b:
            raise ValueError("bundles_a and bundles_b must be non-empty.")

        n_a, n_b = len(bundles_a), len(bundles_b)
        metric_keys = list(_METRIC_REGISTRY.keys()) if metrics is None else metrics
        is_multi_run = (n_a > 1 and n_b > 1)

        if is_multi_run and n_a != n_b:
            raise ValueError(
                f"For multi-run paired tests, bundles_a and bundles_b must have the same "
                f"length.  Got n_a={n_a}, n_b={n_b}."
            )

        report = SignificanceReport(
            model_a=model_a,
            model_b=model_b,
            n_runs_a=n_a,
            n_runs_b=n_b,
            alpha=self.alpha,
            correction=correction,
        )

        # ── Extract scalar metric values ──────────────────────────────────────
        results_a: Dict[str, List[float]] = {}
        results_b: Dict[str, List[float]] = {}

        for key in metric_keys:
            if key not in _METRIC_REGISTRY:
                logger.warning("SignificanceTester: unknown metric '%s'; skipped.", key)
                continue
            path, _, _ = _METRIC_REGISTRY[key]
            vals_a = [self._extract(b, path) for b in bundles_a]
            vals_b = [self._extract(b, path) for b in bundles_b]
            results_a[key] = vals_a
            results_b[key] = vals_b

        # ── Run per-metric tests ──────────────────────────────────────────────
        raw_p: List[float] = []
        metric_order: List[str] = []
        partial_results: Dict[str, MetricTestResult] = {}

        for key in results_a:
            _, higher_is_better, ci_path = _METRIC_REGISTRY[key]
            vals_a_arr = np.array(results_a[key], dtype=float)
            vals_b_arr = np.array(results_b[key], dtype=float)

            mean_a = float(np.nanmean(vals_a_arr))
            mean_b = float(np.nanmean(vals_b_arr))
            diff = mean_b - mean_a

            if is_multi_run:
                mr = self._multi_run_test(
                    key, vals_a_arr, vals_b_arr, mean_a, mean_b, diff, higher_is_better
                )
            else:
                # Single-run: use CI overlap
                ci_a = self._extract_ci(bundles_a[0], ci_path) if ci_path else None
                ci_b = self._extract_ci(bundles_b[0], ci_path) if ci_path else None
                mr = self._single_run_test(
                    key, mean_a, mean_b, diff, ci_a, ci_b, higher_is_better
                )

            partial_results[key] = mr
            metric_order.append(key)
            raw_p.append(mr.p_value)

        # ── Apply Holm–Bonferroni correction ──────────────────────────────────
        if correction == "holm":
            adj_p = _holm_bonferroni(raw_p, self.alpha)
        else:
            adj_p = raw_p  # fallback: no correction

        for i, key in enumerate(metric_order):
            mr = partial_results[key]
            mr.adjusted_p_value = adj_p[i]
            # Re-evaluate significance using adjusted p (multi-run only)
            if mr.test_method in ("wilcoxon", "paired_t") and not np.isnan(adj_p[i]):
                mr.significant = adj_p[i] < self.alpha
            report.results[key] = mr

        return report

    # ── Multi-run tests ───────────────────────────────────────────────────────

    def _multi_run_test(
        self,
        key: str,
        vals_a: np.ndarray,
        vals_b: np.ndarray,
        mean_a: float,
        mean_b: float,
        diff: float,
        higher_is_better: bool,
    ) -> MetricTestResult:
        """Wilcoxon (k≥5) or paired-t (k=2..4) on matched run arrays."""
        from scipy import stats

        # Drop NaN pairs
        valid = ~(np.isnan(vals_a) | np.isnan(vals_b))
        a_v = vals_a[valid]
        b_v = vals_b[valid]
        n = len(a_v)

        if n < 2:
            return MetricTestResult(
                metric=key, value_a=mean_a, value_b=mean_b, difference=diff,
                test_method="na",
            )

        differences = b_v - a_v

        if n >= 5:
            # Wilcoxon signed-rank test
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stat, p = stats.wilcoxon(differences, alternative="two-sided")
                method = "wilcoxon"
                # Rank-biserial correlation r = 1 − 2W / (n(n+1)/2)
                # W here is the Wilcoxon statistic (sum of signed ranks)
                # scipy returns the minimum of W+ and W-; use the formula:
                # r_rb = 1 - (2 * W_stat) / (n*(n+1)/2)
                w_max = n * (n + 1) / 2.0
                r_rb = float(1.0 - 2.0 * stat / w_max) if w_max > 0 else _NAN
                effect_size = r_rb
            except Exception as exc:
                logger.warning("SignificanceTester Wilcoxon failed for '%s': %s", key, exc)
                return MetricTestResult(
                    metric=key, value_a=mean_a, value_b=mean_b, difference=diff,
                    test_method="na",
                )
        else:
            # Paired t-test
            try:
                stat, p = stats.ttest_rel(a_v, b_v, alternative="two-sided")
                method = "paired_t"
                # Cohen's d for paired samples: mean(diff) / std(diff)
                std_d = float(np.std(differences, ddof=1))
                cohens_d = float(np.mean(differences) / std_d) if std_d > 0 else _NAN
                effect_size = cohens_d
            except Exception as exc:
                logger.warning("SignificanceTester paired-t failed for '%s': %s", key, exc)
                return MetricTestResult(
                    metric=key, value_a=mean_a, value_b=mean_b, difference=diff,
                    test_method="na",
                )

        return MetricTestResult(
            metric=key,
            value_a=mean_a,
            value_b=mean_b,
            difference=diff,
            test_method=method,
            p_value=float(p),
            effect_size=effect_size,
        )

    # ── Single-run CI-overlap test ────────────────────────────────────────────

    def _single_run_test(
        self,
        key: str,
        mean_a: float,
        mean_b: float,
        diff: float,
        ci_a: Optional[Tuple[float, float]],
        ci_b: Optional[Tuple[float, float]],
        higher_is_better: bool,
    ) -> MetricTestResult:
        """CI-overlap significance proxy for single-run comparisons."""
        if ci_a is None or ci_b is None:
            return MetricTestResult(
                metric=key, value_a=mean_a, value_b=mean_b, difference=diff,
                test_method="na",
            )

        lo_a, hi_a = ci_a
        lo_b, hi_b = ci_b

        if any(np.isnan(x) for x in [lo_a, hi_a, lo_b, hi_b]):
            return MetricTestResult(
                metric=key, value_a=mean_a, value_b=mean_b, difference=diff,
                test_method="ci_overlap",
            )

        # CIs overlap when max(lo_a, lo_b) < min(hi_a, hi_b)
        overlaps = max(lo_a, lo_b) < min(hi_a, hi_b)
        significant = not overlaps

        return MetricTestResult(
            metric=key,
            value_a=mean_a,
            value_b=mean_b,
            difference=diff,
            test_method="ci_overlap",
            significant=significant,
            ci_overlap=overlaps,
        )

    # ── Extraction helpers ────────────────────────────────────────────────────

    @staticmethod
    def _extract(bundle: Any, attr_path: str) -> float:
        """Extract a scalar metric value from a ResultBundle via dot notation.

        For dict-valued attributes (Dim3, Dim5), returns the mean of all
        non-NaN numeric values in the dict.
        For tuple-valued attributes (CI), returns the mean of lo and hi.

        Returns NaN on any attribute access or type error.
        """
        try:
            obj = bundle
            parts = attr_path.split(".")
            for part in parts:
                if obj is None:
                    return _NAN
                obj = getattr(obj, part, None)

            if obj is None:
                return _NAN

            if isinstance(obj, float):
                return obj
            if isinstance(obj, (int,)):
                return float(obj)
            if isinstance(obj, dict):
                vals = [v for v in obj.values() if isinstance(v, (int, float)) and not np.isnan(v)]
                return float(np.mean(vals)) if vals else _NAN
            if isinstance(obj, (tuple, list)) and len(obj) == 2:
                lo, hi = obj
                if lo is not None and hi is not None:
                    return float((lo + hi) / 2.0)
            return _NAN
        except Exception:
            return _NAN

    @staticmethod
    def _extract_ci(
        bundle: Any, ci_path: str
    ) -> Optional[Tuple[float, float]]:
        """Extract a CI tuple from a ResultBundle.

        For ``Dict[str, Tuple[float, float]]`` CI attributes (Dim5), returns
        the element-wise mean CI: (mean_lo, mean_hi).
        """
        try:
            obj = bundle
            for part in ci_path.split("."):
                if obj is None:
                    return None
                obj = getattr(obj, part, None)

            if obj is None:
                return None
            if isinstance(obj, tuple) and len(obj) == 2:
                lo, hi = float(obj[0]), float(obj[1])
                if np.isnan(lo) or np.isnan(hi):
                    return None
                return (lo, hi)
            if isinstance(obj, dict):
                # Dict[str, Tuple[float, float]] — average lo and hi
                los, his = [], []
                for v in obj.values():
                    if isinstance(v, tuple) and len(v) == 2:
                        lo, hi = v
                        if not (np.isnan(lo) or np.isnan(hi)):
                            los.append(lo)
                            his.append(hi)
                if los and his:
                    return (float(np.mean(los)), float(np.mean(his)))
            return None
        except Exception:
            return None

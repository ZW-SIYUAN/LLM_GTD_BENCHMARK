"""
Visualization & aggregation layer for LLM-GTD-Benchmark results.

Provides three publication-quality figure types via :class:`ResultAggregator`:

    :meth:`~ResultAggregator.plot_radar`
        Six-axis holistic radar chart — one axis per evaluation dimension.
        Normalisation: ``"minmax"`` (relative to the current model set) or
        ``"baseline"`` (relative to a named reference model).

    :meth:`~ResultAggregator.to_leaderboard`
        Dense DataFrame with all key metrics, optional bootstrap confidence
        intervals (computed across multiple evaluation runs per model), and a
        weighted composite score column.

    :meth:`~ResultAggregator.plot_pareto` / :meth:`~ResultAggregator.plot_trade_offs`
        2D Pareto-frontier scatter plots.  Three canonical trade-off panels:

            A  Utility × Privacy   — Macro-F1 vs DCR 5th percentile
            B  Utility × Fairness  — Macro-F1 vs 1 − ΔEO
            C  Privacy × Fairness  — DCR 5th   vs 1 − ΔEO

        Panel C reveals the privacy–fairness tension (Pujol et al. 2020):
        models with higher DCR may under-represent minority groups, raising ΔEO.

Design principles
-----------------
- ``add_model()`` can be called multiple times for the same model name to
  register multiple evaluation runs.  Bootstrap CI is computed automatically
  when ≥ 2 runs are present.
- DCR (Privacy axis) cannot be directly placed on a [0,1] radar because its
  absolute scale is dataset-dependent.  Pass ``dcr_reference`` (the real
  holdout self-DCR) for absolute normalisation; otherwise the axis is
  min-max scaled within the current model set (with a UserWarning).
- ``to_leaderboard()`` works without matplotlib; only plotting methods
  require it (``pip install matplotlib``).
- No existing evaluation code (dimension0-5.py) is modified.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from llm_gtd_benchmark.metrics.dimension0 import Dim0Result
from llm_gtd_benchmark.metrics.dimension1 import Dim1Result
from llm_gtd_benchmark.metrics.dimension2 import Dim2Result
from llm_gtd_benchmark.metrics.dimension3 import Dim3Result
from llm_gtd_benchmark.metrics.dimension4 import Dim4Result
from llm_gtd_benchmark.metrics.dimension5 import Dim5Result

logger = logging.getLogger(__name__)

_NAN = float("nan")
_REGRESSION_VAL = "regression"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default composite score weights (must sum to 1.0)
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "structural": 0.15,
    "fidelity":   0.20,
    "logic":      0.20,
    "utility":    0.20,
    "privacy":    0.15,
    "fairness":   0.10,
}

# Radar axes: (internal key, display label)
_RADAR_AXES: List[Tuple[str, str]] = [
    ("structural", "Structural\nValidity"),
    ("fidelity",   "Distributional\nFidelity"),
    ("logic",      "Logic\nConsistency"),
    ("utility",    "Downstream\nUtility"),
    ("privacy",    "Privacy\nProtection"),
    ("fairness",   "Fairness\n1 − ΔEO"),
]

# Pareto metric registry: key → (axis label, raw_score_key, higher_is_better)
_PARETO_METRICS: Dict[str, Tuple[str, str, bool]] = {
    "utility":      ("Best TSTR F1 / R²",  "utility_tstr_f1",  True),
    "privacy_dcr":  ("DCR 5th Percentile", "dcr_5th",          True),
    "fairness_eo":  ("Mean ΔEO",           "delta_eo_mean",    False),
    "fidelity":     ("1 − Mean KS",        "fidelity_score",   True),
    "structural":   ("1 − IRR",            "structural_score", True),
}

# Canonical three trade-off panels
_TRADE_OFF_PANELS: List[Tuple[str, str, str]] = [
    ("utility",     "privacy_dcr", "Utility × Privacy"),
    ("utility",     "fairness_eo", "Utility × Fairness"),
    ("privacy_dcr", "fairness_eo", "Privacy × Fairness"),
]

# Publication colour palette (colourblind-friendly)
_COLORS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#795548", "#607D8B",
    "#E91E63", "#009688",
]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _safe(obj: Any, attr: str, default: float = _NAN) -> float:
    """Safely read a float attribute; return default on missing/NaN/None."""
    v = getattr(obj, attr, default)
    if v is None:
        return default
    try:
        f = float(v)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _mean_valid(values: List[float]) -> float:
    """Arithmetic mean of non-NaN values; NaN when none exist."""
    valid = [v for v in values if not np.isnan(v)]
    return float(np.mean(valid)) if valid else _NAN


def _best_model_metric(
    model_dict: Optional[Dict[str, Dict[str, float]]],
    key: str,
    higher: bool = True,
) -> float:
    """Best (max or min) value of *key* across all model entries in *model_dict*."""
    if not model_dict:
        return _NAN
    vals = [
        v for scores in model_dict.values()
        if not np.isnan(v := scores.get(key, _NAN))
    ]
    if not vals:
        return _NAN
    return float(np.max(vals) if higher else np.min(vals))


def _bootstrap_ci(
    values: List[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap CI of the mean across runs.  Returns (lower, upper).

    Requires ≥ 2 non-NaN values; returns (NaN, NaN) otherwise.
    """
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if len(arr) < 2:
        return _NAN, _NAN
    rng = np.random.RandomState(seed)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(boot_means, [alpha * 100, (1 - alpha) * 100])
    return float(lo), float(hi)


def _pareto_mask(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points (maximise both axes).

    A point p is Pareto-optimal when no other point q satisfies
    xs[q] ≥ xs[p], ys[q] ≥ ys[p], with at least one strict inequality.
    """
    n = len(xs)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if xs[j] >= xs[i] and ys[j] >= ys[i] and (xs[j] > xs[i] or ys[j] > ys[i]):
                is_pareto[i] = False
                break
    return is_pareto


def _require_matplotlib() -> None:
    """Raise ImportError with install hint when matplotlib is absent."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise ImportError(
            "Plotting methods require matplotlib.  "
            "Install with:  pip install matplotlib"
        ) from None


# ---------------------------------------------------------------------------
# Data container — one evaluation run
# ---------------------------------------------------------------------------


@dataclass
class _RunEntry:
    """All DimN results for a single evaluation run of one model."""

    result0: Optional[Dim0Result] = None
    result1: Optional[Dim1Result] = None
    result2: Optional[Dim2Result] = None
    result3: Optional[Dim3Result] = None
    result4: Optional[Dim4Result] = None
    result5: Optional[Dim5Result] = None


# ---------------------------------------------------------------------------
# ResultAggregator
# ---------------------------------------------------------------------------


class ResultAggregator:
    """Aggregate multi-model benchmark results into publication-quality figures.

    Parameters
    ----------
    baseline_model:
        Name of the reference model for ``normalize="baseline"`` radar charts.
        Must match a name previously passed to :meth:`add_model`.
    dcr_reference:
        Optional real-data self-DCR value — the 5th-percentile distance from a
        real held-out set to the real training set.  Provides a dataset-absolute
        scale for the Privacy radar axis.  When ``None``, the maximum DCR across
        all added models is used and a ``UserWarning`` is raised.
    composite_weights:
        Per-axis weights for the leaderboard composite score.  Must sum to 1.
        Defaults to ``{structural:0.15, fidelity:0.20, logic:0.20,
        utility:0.20, privacy:0.15, fairness:0.10}``.

    Examples
    --------
    >>> agg = ResultAggregator(baseline_model="CTGAN")
    >>> agg.add_model("GPT-4",   result0=r0, result1=r1, result3=r3,
    ...                          result4=r4, result5=r5)
    >>> agg.add_model("LLaMA-3", result0=r0b, result1=r1b, result3=r3b,
    ...                          result4=r4b, result5=r5b)
    >>> agg.add_model("CTGAN",   result0=r0c, result1=r1c, result3=r3c,
    ...                          result4=r4c, result5=r5c)
    >>>
    >>> fig_radar = agg.plot_radar()
    >>> df_board  = agg.to_leaderboard(compact=True)
    >>> fig_trade = agg.plot_trade_offs()
    """

    def __init__(
        self,
        baseline_model: str,
        dcr_reference: Optional[float] = None,
        composite_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.baseline_model = baseline_model
        self.dcr_reference = dcr_reference
        self._weights = composite_weights or dict(_DEFAULT_WEIGHTS)

        if abs(sum(self._weights.values()) - 1.0) > 1e-6:
            raise ValueError(
                f"composite_weights must sum to 1.0; got {sum(self._weights.values()):.6f}"
            )
        if set(self._weights) != {ax for ax, _ in _RADAR_AXES}:
            raise ValueError(
                f"composite_weights keys must be exactly: {[ax for ax, _ in _RADAR_AXES]}"
            )

        self._runs: Dict[str, List[_RunEntry]] = {}
        self._order: List[str] = []  # insertion order preserved

    # ── Public: add model ────────────────────────────────────────────────────

    def add_model(
        self,
        name: str,
        result0: Optional[Dim0Result] = None,
        result1: Optional[Dim1Result] = None,
        result2: Optional[Dim2Result] = None,
        result3: Optional[Dim3Result] = None,
        result4: Optional[Dim4Result] = None,
        result5: Optional[Dim5Result] = None,
    ) -> None:
        """Register one evaluation run for *name*.

        Calling this method multiple times with the same *name* adds multiple
        runs; confidence intervals in :meth:`to_leaderboard` are then computed
        via bootstrap resampling across runs.

        Parameters
        ----------
        name:
            Human-readable model identifier (e.g. ``"GPT-4-turbo"``).
        result0 … result5:
            Corresponding :class:`~llm_gtd_benchmark.metrics.DimNResult`
            objects.  Any can be ``None`` when a dimension was not evaluated.
        """
        entry = _RunEntry(result0, result1, result2, result3, result4, result5)
        if name not in self._runs:
            self._runs[name] = []
            self._order.append(name)
        self._runs[name].append(entry)
        logger.debug(
            "ResultAggregator: registered run #%d for '%s'.",
            len(self._runs[name]),
            name,
        )

    # ── Public: leaderboard ──────────────────────────────────────────────────

    def to_leaderboard(
        self,
        n_boot: int = 1000,
        ci: float = 0.95,
        compact: bool = False,
    ) -> pd.DataFrame:
        """Produce a leaderboard DataFrame with all key metrics.

        Parameters
        ----------
        n_boot:
            Bootstrap resamples for CI computation (only active when ≥ 2 runs
            per model).  Default: 1000.
        ci:
            Confidence level.  Default: 0.95.
        compact:
            ``True`` → return a 7-column paper-friendly view
            (one representative metric per dimension + composite score).
            ``False`` (default) → return all extracted metrics.

        Returns
        -------
        pd.DataFrame
            Rows = models (sorted by composite score descending).
            Numeric cells contain mean values; CI bounds are stored in
            ``{metric}_ci_lo`` / ``{metric}_ci_hi`` columns (full mode only).
        """
        self._check_nonempty()

        records: Dict[str, Dict[str, Any]] = {}
        for name in self._order:
            raw_per_run = [self._extract_raw_scores(e) for e in self._runs[name]]
            records[name] = self._aggregate_with_ci(raw_per_run, n_boot, ci)

        df = pd.DataFrame.from_dict(records, orient="index")
        df.index.name = "Model"

        # Composite score (weighted sum of normalised radar axes)
        radar_per_model = {
            name: [self._radar_raw_scores(e) for e in self._runs[name]]
            for name in self._order
        }
        norm = self._normalize_radar_all(radar_per_model)
        composite = {}
        for name in self._order:
            axis_scores = norm[name]
            composite[name] = float(
                sum(
                    self._weights[ax] * (v if not np.isnan(v) else 0.0)
                    for ax in self._weights
                    for v in (axis_scores.get(ax, _NAN),)
                )
            )
        df.insert(0, "Score", pd.Series(composite))

        df.sort_values("Score", ascending=False, inplace=True)

        if compact:
            _COMPACT_COLS = {
                "irr":             "IRR (↓)",
                "mean_ks":         "KS (↓)",
                "dsi_gap":         "DSI Gap (↓)",
                "utility_tstr_f1": "TSTR F1 (↑)",
                "dcr_5th":         "DCR 5th (↑)",
                "delta_eo_mean":   "ΔEO (↓)",
            }
            available = ["Score"] + [c for c in _COMPACT_COLS if c in df.columns]
            df = df[available].rename(columns=_COMPACT_COLS)

        return df

    # ── Public: radar chart ──────────────────────────────────────────────────

    def plot_radar(
        self,
        normalize: str = "minmax",
        title: str = "LLM-GTD Benchmark — Holistic Radar",
        figsize: Tuple[float, float] = (7, 7),
        alpha_fill: float = 0.15,
    ) -> Any:
        """Six-axis holistic radar chart.

        Parameters
        ----------
        normalize:
            ``"minmax"`` — scale each axis to [0, 1] within the current model
            set (axis spans depend on which models are present).
            ``"baseline"`` — normalise each axis relative to
            ``self.baseline_model`` (score 1.0 = equal to baseline).
        title:
            Figure title.
        figsize:
            Matplotlib figure size.
        alpha_fill:
            Opacity of the filled polygon for each model.

        Returns
        -------
        ``matplotlib.figure.Figure``
        """
        _require_matplotlib()
        import matplotlib.pyplot as plt

        self._check_nonempty()

        radar_per_model = {
            name: [self._radar_raw_scores(e) for e in self._runs[name]]
            for name in self._order
        }
        norm = self._normalize_radar_all(radar_per_model, strategy=normalize)

        n_axes = len(_RADAR_AXES)
        angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
        angles_closed = angles + angles[:1]

        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_xticks(angles)
        ax.set_xticklabels([lbl for _, lbl in _RADAR_AXES], fontsize=9.5, fontweight="bold")
        ax.set_rlabel_position(10)
        ax.set_yticks([0.25, 0.50, 0.75, 1.00])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="#888")
        ax.set_ylim(0, 1.08)

        for idx, name in enumerate(self._order):
            color = _COLORS[idx % len(_COLORS)]
            scores = norm[name]
            values = [max(scores.get(k, 0.0) or 0.0, 0.0) for k, _ in _RADAR_AXES]
            values_closed = values + values[:1]

            ax.plot(angles_closed, values_closed, color=color, linewidth=2.2, label=name)
            ax.fill(angles_closed, values_closed, color=color, alpha=alpha_fill)

            # Mark missing axes with ✕
            for i, (ax_key, _) in enumerate(_RADAR_AXES):
                if np.isnan(scores.get(ax_key, _NAN)):
                    ax.plot(angles[i], 0.0, "x", color=color, markersize=9, zorder=4)

        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.42, 1.18),
            fontsize=9,
            framealpha=0.9,
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=22)
        fig.tight_layout()
        return fig

    # ── Public: single Pareto plot ────────────────────────────────────────────

    def plot_pareto(
        self,
        x_metric: str,
        y_metric: str,
        title: str = "",
        figsize: Tuple[float, float] = (6, 5),
        annotate: bool = True,
    ) -> Any:
        """2D Pareto-frontier scatter plot for two metrics.

        Parameters
        ----------
        x_metric, y_metric:
            Metric keys.  Supported values:
            ``"utility"``, ``"privacy_dcr"``, ``"fairness_eo"``,
            ``"fidelity"``, ``"structural"``.
        title:
            Axes title.  Auto-generated when empty.
        figsize:
            Figure size.
        annotate:
            Label each point with the model name.

        Returns
        -------
        ``matplotlib.figure.Figure``
        """
        _require_matplotlib()
        import matplotlib.pyplot as plt

        self._check_pareto_metrics(x_metric, y_metric)
        fig, ax = plt.subplots(figsize=figsize)
        self._draw_pareto_on_ax(ax, x_metric, y_metric, title=title, annotate=annotate)
        fig.tight_layout()
        return fig

    # ── Public: 3-panel trade-off figure ─────────────────────────────────────

    def plot_trade_offs(
        self,
        figsize: Tuple[float, float] = (16, 5),
        suptitle: str = "LLM-GTD Benchmark — Trade-off Pareto Frontiers",
    ) -> Any:
        """Three-panel Pareto figure: utility×privacy, utility×fairness, privacy×fairness.

        The third panel (Privacy × Fairness) is often absent in related work
        but reveals a critical tension: models with high DCR (strong privacy)
        may under-represent minority groups, driving up ΔEO (Pujol et al. 2020).

        Returns
        -------
        ``matplotlib.figure.Figure``
        """
        _require_matplotlib()
        import matplotlib.pyplot as plt

        self._check_nonempty()

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        for ax, (xm, ym, panel_title) in zip(axes, _TRADE_OFF_PANELS):
            self._draw_pareto_on_ax(ax, xm, ym, title=panel_title, annotate=True)

        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        return fig

    # ── Private: metric extraction ────────────────────────────────────────────

    def _extract_raw_scores(self, entry: _RunEntry) -> Dict[str, float]:
        """Extract a flat dict of all raw metric values from one run entry.

        All values are floats; NaN when not available.
        """
        r0 = entry.result0
        r1 = entry.result1
        r2 = entry.result2
        r3 = entry.result3
        r4 = entry.result4
        r5 = entry.result5

        s: Dict[str, float] = {}

        # ── Dim0 ──────────────────────────────────────────────────────────────
        irr = _safe(r0, "irr")
        s["irr"] = irr
        s["structural_score"] = 1.0 - irr if not np.isnan(irr) else _NAN

        # ── Dim1 ──────────────────────────────────────────────────────────────
        ks = _safe(r1, "mean_ks")
        s["mean_ks"] = ks
        s["mean_tvd"] = _safe(r1, "mean_tvd")
        s["alpha_precision"] = _safe(r1, "alpha_precision")
        s["beta_recall"] = _safe(r1, "beta_recall")
        c2st_xgb = _safe(r1, "c2st_auc_xgb")
        c2st_rf = _safe(r1, "c2st_auc_rf")
        # C2ST: closer to 0.5 = better → represent as |AUC - 0.5| (lower=better)
        c2st_dev = _mean_valid([
            abs(c2st_xgb - 0.5) if not np.isnan(c2st_xgb) else _NAN,
            abs(c2st_rf - 0.5) if not np.isnan(c2st_rf) else _NAN,
        ])
        s["c2st_deviation_mean"] = c2st_dev
        s["fidelity_score"] = 1.0 - ks if not np.isnan(ks) else _NAN

        # ── Dim2 ──────────────────────────────────────────────────────────────
        s["dsi_gap"] = _safe(r2, "dsi_gap")
        s["dsi_synth_ll"] = _safe(r2, "dsi_synth_ll")
        s["icvr"] = _safe(r2, "icvr")
        s["hcs_violation_rate"] = _safe(r2, "hcs_violation_rate")
        s["mdi_mean"] = _safe(r2, "mdi_mean")

        # ── Dim3 ──────────────────────────────────────────────────────────────
        is_reg = r3 is not None and _REGRESSION_VAL in getattr(r3, "task_type", "")
        mle_tstr = getattr(r3, "mle_tstr", None)
        mle_trtr = getattr(r3, "mle_trtr", None)
        utility_gap = getattr(r3, "utility_gap", None)

        if is_reg:
            s["utility_tstr_f1"] = _best_model_metric(mle_tstr, "r2",     higher=True)
            s["utility_trtr_f1"] = _best_model_metric(mle_trtr, "r2",     higher=True)
            s["utility_tstr_auc"] = _NAN
            s["utility_gap_best"] = _best_model_metric(utility_gap, "r2", higher=False)
        else:
            s["utility_tstr_f1"] = _best_model_metric(mle_tstr, "macro_f1",     higher=True)
            s["utility_trtr_f1"] = _best_model_metric(mle_trtr, "macro_f1",     higher=True)
            s["utility_tstr_auc"] = _best_model_metric(mle_tstr, "roc_auc",     higher=True)
            s["utility_gap_best"] = _best_model_metric(utility_gap, "macro_f1", higher=False)

        # ── Dim4 ──────────────────────────────────────────────────────────────
        s["dcr_5th"] = _safe(r4, "dcr_5th_percentile")
        s["exact_match_rate"] = _safe(r4, "exact_match_rate")
        s["dlt_gap"] = _safe(r4, "dlt_gap")

        # ── Dim5 ──────────────────────────────────────────────────────────────
        def _mean_dim5_dict(attr: str) -> float:
            d = getattr(r5, attr, None) if r5 else None
            if not d:
                return _NAN
            return _mean_valid(list(d.values()))

        s["delta_dp_mean"] = _mean_dim5_dict("delta_dp")
        s["delta_eop_mean"] = _mean_dim5_dict("delta_eop")
        s["delta_eo_mean"] = _mean_dim5_dict("delta_eo")
        s["stat_parity_gap_mean"] = _mean_dim5_dict("stat_parity_gap")

        # NMI bias amplification (mean Δ = synth − real)
        nmi_real = getattr(r5, "bias_nmi_real", {}) if r5 else {}
        nmi_synth = getattr(r5, "bias_nmi_synth", {}) if r5 else {}
        deltas = [
            nmi_synth[c] - nmi_real[c]
            for c in nmi_real
            if not np.isnan(nmi_synth.get(c, _NAN)) and not np.isnan(nmi_real[c])
        ]
        s["nmi_delta_mean"] = _mean_valid(deltas) if deltas else _NAN

        return s

    def _radar_raw_scores(self, entry: _RunEntry) -> Dict[str, float]:
        """Compute 6 radar-axis scores (all *higher = better*) for one run.

        Returned scores are un-normalised.  The Privacy axis retains the raw
        DCR value; normalisation is performed at plot time.
        """
        raw = self._extract_raw_scores(entry)

        dsi_gap = raw.get("dsi_gap", _NAN)
        logic = 1.0 / (1.0 + dsi_gap) if not np.isnan(dsi_gap) else _NAN

        delta_eo_mean = raw.get("delta_eo_mean", _NAN)
        fairness = 1.0 - delta_eo_mean if not np.isnan(delta_eo_mean) else _NAN

        return {
            "structural": raw.get("structural_score", _NAN),
            "fidelity":   raw.get("fidelity_score", _NAN),
            "logic":      logic,
            "utility":    raw.get("utility_tstr_f1", _NAN),
            "privacy":    raw.get("dcr_5th", _NAN),   # normalised later
            "fairness":   fairness,
        }

    # ── Private: normalisation ────────────────────────────────────────────────

    def _normalize_radar_all(
        self,
        radar_per_model: Dict[str, List[Dict[str, float]]],
        strategy: str = "minmax",
    ) -> Dict[str, Dict[str, float]]:
        """Normalise all six radar axes to [0, 1] across the model set.

        1. Average across runs for each model.
        2. For the Privacy axis: scale by ``dcr_reference`` if provided,
           otherwise by the maximum DCR across models (+ UserWarning).
        3. For remaining axes:
           - ``"baseline"``: divide by baseline model's score (cap at 1.5, then /1.5).
           - ``"minmax"``: standard min-max across the model set.

        Returns ``Dict[model_name, Dict[axis_key, normalised_score ∈ [0,1]]]``.
        """
        # Step 1: mean across runs
        mean_scores: Dict[str, Dict[str, float]] = {}
        for name, run_list in radar_per_model.items():
            mean_scores[name] = {}
            for ax_key, _ in _RADAR_AXES:
                vals = [r.get(ax_key, _NAN) for r in run_list]
                mean_scores[name][ax_key] = _mean_valid(vals)

        normalised: Dict[str, Dict[str, float]] = {n: {} for n in mean_scores}

        for ax_key, _ in _RADAR_AXES:
            all_vals = [mean_scores[n][ax_key] for n in mean_scores]
            valid = [v for v in all_vals if not np.isnan(v)]

            if not valid:
                for n in mean_scores:
                    normalised[n][ax_key] = _NAN
                continue

            # ── Privacy: always normalise by external reference or max ───────
            if ax_key == "privacy":
                ref = self.dcr_reference
                if ref is None:
                    ref = float(np.max(valid))
                    if ref == 0.0:
                        ref = 1.0
                    warnings.warn(
                        "ResultAggregator: dcr_reference not provided; "
                        "Privacy radar axis normalised by max DCR across models. "
                        "Results are relative, not absolute.  "
                        "Pass dcr_reference for dataset-absolute scores.",
                        UserWarning,
                        stacklevel=3,
                    )
                for n in mean_scores:
                    v = mean_scores[n]["privacy"]
                    normalised[n]["privacy"] = (
                        float(np.clip(v / ref, 0.0, 1.0))
                        if not np.isnan(v)
                        else _NAN
                    )
                continue

            # ── Other axes: baseline or minmax ───────────────────────────────
            effective_strategy = strategy
            if effective_strategy == "baseline":
                ref_name = self.baseline_model
                if ref_name not in mean_scores:
                    warnings.warn(
                        f"baseline_model='{ref_name}' not in model set; "
                        "falling back to minmax.",
                        UserWarning,
                        stacklevel=3,
                    )
                    effective_strategy = "minmax"
                else:
                    ref_val = mean_scores[ref_name][ax_key]
                    if np.isnan(ref_val) or ref_val == 0.0:
                        effective_strategy = "minmax"
                    else:
                        for n in mean_scores:
                            v = mean_scores[n][ax_key]
                            normalised[n][ax_key] = (
                                float(np.clip(v / ref_val, 0.0, 1.5) / 1.5)
                                if not np.isnan(v)
                                else _NAN
                            )
                        continue  # next axis

            # minmax
            vmin, vmax = float(np.min(valid)), float(np.max(valid))
            span = vmax - vmin if vmax > vmin else 1.0
            for n in mean_scores:
                v = mean_scores[n][ax_key]
                normalised[n][ax_key] = (
                    float(np.clip((v - vmin) / span, 0.0, 1.0))
                    if not np.isnan(v)
                    else _NAN
                )

        return normalised

    # ── Private: CI aggregation ───────────────────────────────────────────────

    def _aggregate_with_ci(
        self,
        raw_per_run: List[Dict[str, float]],
        n_boot: int,
        ci: float,
    ) -> Dict[str, Any]:
        """Aggregate raw score dicts across runs; compute bootstrap CI when ≥ 2 runs."""
        all_keys = sorted(set().union(*[r.keys() for r in raw_per_run]))
        result: Dict[str, Any] = {}

        for key in all_keys:
            vals = [r.get(key, _NAN) for r in raw_per_run]
            valid = [v for v in vals if not np.isnan(v)]

            if not valid:
                result[key] = _NAN
            elif len(valid) == 1:
                result[key] = valid[0]
            else:
                result[key] = float(np.mean(valid))
                lo, hi = _bootstrap_ci(valid, n_boot=n_boot, ci=ci)
                result[f"{key}_ci_lo"] = lo
                result[f"{key}_ci_hi"] = hi

        return result

    # ── Private: Pareto drawing ────────────────────────────────────────────────

    def _draw_pareto_on_ax(
        self,
        ax: Any,
        x_metric: str,
        y_metric: str,
        title: str = "",
        annotate: bool = True,
    ) -> None:
        """Draw a Pareto scatter + frontier onto an existing matplotlib Axes."""
        self._check_pareto_metrics(x_metric, y_metric)

        x_label, x_key, x_higher = _PARETO_METRICS[x_metric]
        y_label, y_key, y_higher = _PARETO_METRICS[y_metric]

        names = list(self._order)
        xs_raw = np.array([
            _mean_valid([self._extract_raw_scores(e).get(x_key, _NAN)
                         for e in self._runs[n]])
            for n in names
        ], dtype=float)
        ys_raw = np.array([
            _mean_valid([self._extract_raw_scores(e).get(y_key, _NAN)
                         for e in self._runs[n]])
            for n in names
        ], dtype=float)

        # Transform to "higher is better" space for Pareto computation
        xs_p = xs_raw if x_higher else -xs_raw
        ys_p = ys_raw if y_higher else -ys_raw

        valid = ~(np.isnan(xs_p) | np.isnan(ys_p))
        pareto_mask_arr = np.zeros(len(names), dtype=bool)
        if valid.sum() >= 2:
            pareto_mask_arr[valid] = _pareto_mask(xs_p[valid], ys_p[valid])

        for idx, name in enumerate(names):
            color = _COLORS[idx % len(_COLORS)]
            xv, yv = xs_raw[idx], ys_raw[idx]
            if np.isnan(xv) or np.isnan(yv):
                continue

            is_on_frontier = pareto_mask_arr[idx]
            marker, ms = ("*", 140) if is_on_frontier else ("o", 70)
            ax.scatter(xv, yv, color=color, s=ms, marker=marker, zorder=3, label=name)

            if annotate:
                ax.annotate(
                    name, (xv, yv),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, color=color, fontweight="semibold",
                )

        # Pareto frontier step function (sorted by x)
        frontier_pts = sorted(
            [(xs_raw[i], ys_raw[i])
             for i in range(len(names))
             if pareto_mask_arr[i] and not (np.isnan(xs_raw[i]) or np.isnan(ys_raw[i]))],
            key=lambda p: p[0],
        )
        if len(frontier_pts) >= 2:
            px = [p[0] for p in frontier_pts]
            py = [p[1] for p in frontier_pts]
            ax.step(px, py, where="post", color="#555", linestyle="--",
                    linewidth=1.6, alpha=0.65, label="Pareto frontier", zorder=2)

        # Soft shading of the "best-of-both" quadrant
        x_mid = np.nanmedian(xs_raw)
        y_mid = np.nanmedian(ys_raw)
        xlim = ax.get_xlim() or (np.nanmin(xs_raw) * 0.9, np.nanmax(xs_raw) * 1.1)
        ylim = ax.get_ylim() or (np.nanmin(ys_raw) * 0.9, np.nanmax(ys_raw) * 1.1)
        best_x_span = (x_mid, xlim[1]) if x_higher else (xlim[0], x_mid)
        best_y_span = (y_mid, ylim[1]) if y_higher else (ylim[0], y_mid)
        ax.axhspan(*best_y_span, facecolor="green", alpha=0.04, zorder=0)
        ax.axvspan(*best_x_span, facecolor="green", alpha=0.04, zorder=0)

        x_dir = "↑ better" if x_higher else "↓ better"
        y_dir = "↑ better" if y_higher else "↓ better"
        ax.set_xlabel(f"{x_label}  [{x_dir}]", fontsize=9)
        ax.set_ylabel(f"{y_label}  [{y_dir}]", fontsize=9)
        ax.set_title(
            title or f"{x_label.split()[0]} × {y_label.split()[0]}",
            fontsize=10, fontweight="bold",
        )
        ax.legend(fontsize=7.5, loc="best", framealpha=0.88)
        ax.grid(True, linestyle=":", alpha=0.45)

    # ── Private: guards ───────────────────────────────────────────────────────

    def _check_nonempty(self) -> None:
        if not self._runs:
            raise RuntimeError(
                "ResultAggregator has no models.  Call add_model() first."
            )

    @staticmethod
    def _check_pareto_metrics(x_metric: str, y_metric: str) -> None:
        supported = list(_PARETO_METRICS)
        if x_metric not in _PARETO_METRICS:
            raise ValueError(
                f"Unknown x_metric '{x_metric}'.  Supported: {supported}"
            )
        if y_metric not in _PARETO_METRICS:
            raise ValueError(
                f"Unknown y_metric '{y_metric}'.  Supported: {supported}"
            )
        if x_metric == y_metric:
            raise ValueError("x_metric and y_metric must be different.")

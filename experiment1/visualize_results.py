"""
LLM-GTD Benchmark — Visualization Script
=========================================
Generates four publication-quality figure sets from benchmark results:

  1. figures/radar_<dataset>.png      — per-dataset radar chart (4 models)
  2. figures/heatmap_<metric>.png     — metrics heatmap (models × datasets)
  3. figures/pareto_income.png        — 3-panel Pareto for income dataset
  4. figures/bar_irr.png              — IRR per model per dataset

Run:
    python visualize_results.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend for saving files
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from llm_gtd_benchmark.core.result_bundle import ResultBundle
from llm_gtd_benchmark.visualization.aggregator import ResultAggregator

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

MODELS   = ["GreaT", "GraFT", "GraDe", "PAFT"]
DATASETS = ["diabetes", "house", "income", "sick", "us_location"]

# Colourblind-friendly palette (one per model)
MODEL_COLORS = {
    "GreaT": "#2196F3",
    "GraFT": "#F44336",
    "GraDe": "#4CAF50",
    "PAFT":  "#FF9800",
}

# ── 0. Load all bundles ────────────────────────────────────────────────────────

bundles: dict = {}
for model in MODELS:
    for ds in DATASETS:
        path = RESULTS_DIR / f"{model}_{ds}.json"
        if path.exists():
            try:
                bundles[(model, ds)] = ResultBundle.load(path)
            except Exception as exc:
                print(f"  [WARN] Could not load {path.name}: {exc}")

print(f"Loaded {len(bundles)} result bundles.")

# ── 1. Radar charts — one per dataset ─────────────────────────────────────────

def make_radar(dataset: str) -> None:
    agg = ResultAggregator(baseline_model="GraDe")
    added = []
    for model in MODELS:
        b = bundles.get((model, dataset))
        if b is None:
            continue
        # Skip fully failed runs (no results at all)
        if all(getattr(b, f"result{i}") is None for i in range(6)):
            continue
        agg.add_model(
            model,
            result0=b.result0,
            result1=b.result1,
            result2=b.result2,
            result3=b.result3,
            result4=b.result4,
            result5=b.result5,
        )
        added.append(model)

    if len(added) < 2:
        print(f"  [SKIP] radar/{dataset}: fewer than 2 models with data.")
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig = agg.plot_radar(
            normalize="minmax",
            title=f"LLM-GTD Benchmark — {dataset.replace('_', ' ').title()}",
            figsize=(7, 7),
        )

    out = FIGURES_DIR / f"radar_{dataset}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


print("\n[1/4] Radar charts …")
for ds in DATASETS:
    make_radar(ds)


# ── 2. Metrics heatmap ────────────────────────────────────────────────────────

print("\n[2/4] Metrics heatmap …")

lb = pd.read_csv(RESULTS_DIR / "leaderboard.csv")

HEATMAP_METRICS = [
    ("irr",              "IRR ↓",             True),    # lower=better → invert
    ("mean_ks",          "Mean KS ↓",          True),
    ("mean_tvd",         "Mean TVD ↓",         True),
    ("alpha_precision",  "α-Precision ↑",      False),
    ("beta_recall",      "β-Recall ↑",         False),
    ("c2st_auc_mean",    "C2ST AUC ↓",         True),
    ("dsi_relative_gap_pct", "DSI Gap % ↓",    True),
    ("mle_tstr_primary", "TSTR Score ↑",       False),
    ("dcr_5th",          "DCR 5th ↑",          False),
    ("dcr_95th",         "DCR 95th ↓",         True),
    ("exact_match_rate", "Exact Match ↓",      True),
]

# Build row label: "Model / Dataset"
lb["label"] = lb["model"].str.strip() + " / " + lb["dataset"]

fig, axes = plt.subplots(
    1, len(HEATMAP_METRICS),
    figsize=(len(HEATMAP_METRICS) * 1.55, len(lb) * 0.42 + 1.6),
)

row_labels = lb["label"].tolist()

for ax, (col, title, lower_better) in zip(axes, HEATMAP_METRICS):
    vals = lb[col].values.astype(float)

    # Normalise to [0, 1] (NaN → kept as NaN)
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    span = vmax - vmin if vmax > vmin else 1.0
    norm_vals = (vals - vmin) / span

    if lower_better:
        norm_vals = 1.0 - norm_vals   # flip so green = good

    mat = norm_vals.reshape(-1, 1)

    cmap = plt.cm.RdYlGn
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    # Annotate cells
    for i, v in enumerate(vals):
        if not np.isnan(v):
            text_color = "black" if 0.3 < norm_vals[i] < 0.75 else "white"
            ax.text(0, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=6.5, color=text_color)
        else:
            ax.text(0, i, "—", ha="center", va="center",
                    fontsize=7, color="#aaa")

    ax.set_xticks([])
    ax.set_title(title, fontsize=7.5, fontweight="bold", rotation=30,
                 ha="left", pad=3)

    if ax is axes[0]:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
    else:
        ax.set_yticks([])

fig.suptitle("LLM-GTD Benchmark — Metrics Overview\n(green = better)",
             fontsize=11, fontweight="bold", y=1.01)
fig.tight_layout(rect=[0, 0, 1, 1])

out = FIGURES_DIR / "heatmap_metrics.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out.name}")


# ── 3. Pareto plots — income dataset ──────────────────────────────────────────

print("\n[3/4] Pareto plots (income dataset) …")

agg_income = ResultAggregator(baseline_model="GraDe")
for model in MODELS:
    b = bundles.get((model, "income"))
    if b is None or b.result0 is None:
        continue
    agg_income.add_model(
        model,
        result0=b.result0, result1=b.result1,
        result2=b.result2, result3=b.result3,
        result4=b.result4, result5=b.result5,
    )

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    fig_pareto = agg_income.plot_trade_offs(
        figsize=(15, 5),
        suptitle="LLM-GTD Benchmark — Trade-off Pareto Frontiers (income dataset)",
    )

out = FIGURES_DIR / "pareto_income.png"
fig_pareto.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig_pareto)
print(f"  Saved {out.name}")


# ── 4. Bar chart — IRR per model per dataset ───────────────────────────────────

print("\n[4/4] IRR bar chart …")

fig, ax = plt.subplots(figsize=(11, 4.5))

ds_labels = DATASETS
n_ds = len(ds_labels)
n_models = len(MODELS)
bar_w = 0.18
offsets = np.arange(n_ds)

for m_idx, model in enumerate(MODELS):
    irr_vals = []
    for ds in ds_labels:
        row = lb[(lb["model"].str.strip() == model) & (lb["dataset"] == ds)]
        if row.empty or pd.isna(row["irr"].values[0]):
            irr_vals.append(0.0)
        else:
            irr_vals.append(float(row["irr"].values[0]))

    x_pos = offsets + (m_idx - (n_models - 1) / 2) * bar_w
    bars = ax.bar(x_pos, irr_vals, width=bar_w,
                  color=MODEL_COLORS[model], label=model,
                  edgecolor="white", linewidth=0.5, alpha=0.88)

    for bar, val in zip(bars, irr_vals):
        if val > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.2%}", ha="center", va="bottom",
                    fontsize=6.5, color=MODEL_COLORS[model], fontweight="bold")

ax.set_xticks(offsets)
ax.set_xticklabels([d.replace("_", "\n") for d in ds_labels], fontsize=9)
ax.set_ylabel("Invalid Row Rate (IRR)  ↓ lower = better", fontsize=9)
ax.set_title("LLM-GTD Benchmark — Invalid Row Rate by Model & Dataset",
             fontsize=11, fontweight="bold", pad=10)
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.set_axisbelow(True)

# Annotate PAFT/house as data-quality failure
paft_house_x = ds_labels.index("house") + (MODELS.index("PAFT") - (n_models - 1) / 2) * bar_w
ax.annotate("data\nfailure", xy=(paft_house_x, 0.01),
            xytext=(paft_house_x + 0.4, 0.08),
            fontsize=6.5, color="#999",
            arrowprops=dict(arrowstyle="->", color="#bbb", lw=0.8))

fig.tight_layout()
out = FIGURES_DIR / "bar_irr.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out.name}")


# ── Summary ────────────────────────────────────────────────────────────────────

print(f"\nAll figures saved to:  {FIGURES_DIR}/")
saved = sorted(FIGURES_DIR.glob("*.png"))
for f in saved:
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name:40s}  {size_kb:>5} KB")

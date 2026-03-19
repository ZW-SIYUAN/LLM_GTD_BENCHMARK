# -*- coding: utf-8 -*-
"""
Experiment 1: Order Quality vs Training-Generation Consistency
==============================================================

Research Question:
  In PAFT-style fixed-order GReaT training, does generation quality improve
  because (H1) the ordering is information-theoretically "good", or because
  (H2) training and generation share the same column ordering (consistency)?

Hypotheses:
  H1 (Quality):     Results correlate with chain conditional entropy score;
                    better orderings yield better synthetic data.
  H2 (Consistency): Any fixed ordering outperforms random-order (GReaT
                    baseline) by similar margins; ordering quality matters
                    little as long as train == generate order.

Design:
  1. Enumerate all 5! = 120 permutations of the 5 features.
  2. Score each via Chain Conditional Entropy (chain_CE):
       CE_chain(order) = sum_{i=0}^{n-2}  H(order[i+1] | order[i])
     Lower chain_CE = each step is more predictable from the prior step.
  3. Select 5 orderings at evenly-spaced quantiles of the score distribution,
     excluding the already-run PAFT_ORDER (cond1b) and PAFT_REVERSED (cond5).
  4. For each: train FixedOrderGReaT + generate starting from order[0].
  5. Load existing cond1a/1b/4/5 results for a full comparison.
  6. Plot chain_CE score vs aggregate quality -> test H1 vs H2.

Usage:
  cd TIDE/
  python exp1_order_quality/run_experiment.py
  python exp1_order_quality/run_experiment.py --epochs 50 --subsample 2000
  python exp1_order_quality/run_experiment.py --conditions ord1 ord3
"""

import argparse
import datetime
import itertools
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import typing as tp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
TIDE_ROOT = HERE.parent
sys.path.insert(0, str(TIDE_ROOT))

from ib_sparse_attention_v07 import IBOrderFinder, FixedOrderGReaT  # noqa: E402
from be_great import GReaT  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("exp1")

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR  = TIDE_ROOT / "data"
TRAIN_CSV = DATA_DIR / "us_location_train.csv"

# Reference orderings already tested in experiment_order_comparison.py
PAFT_ORDER    = ["lon", "bird", "state_code", "lat", "lat_zone"]
PAFT_REVERSED = ["lat_zone", "lat", "state_code", "bird", "lon"]

CAT_COLS = ["state_code", "bird", "lat_zone"]
NUM_COLS = ["lat", "lon"]

# Output dir for previous cond experiment (for loading existing results)
PREV_OUTPUT_DIR = TIDE_ROOT / "output_order_comparison"


# ==============================================================================
# Scoring utilities
# ==============================================================================

def chain_ce_score(
    order: tp.List[str],
    feature_names: tp.List[str],
    CE: np.ndarray,
) -> float:
    """
    Chain Conditional Entropy score (lower = better ordering).

    CE_chain = sum_{i=0}^{n-2}  H(order[i+1] | order[i])

    Uses the pairwise conditional entropy matrix from IBOrderFinder:
      CE[a, b] = H(X_b | X_a)

    Note: This is a first-order Markov approximation. The full sum
    sum H(order[i] | order[0],...,order[i-1]) = H(joint) = constant
    regardless of ordering, so it cannot discriminate orderings.
    The pairwise chain_CE is NOT constant and measures how well each
    feature is predicted by its immediately preceding feature.
    """
    idx = [feature_names.index(f) for f in order]
    return float(sum(CE[idx[i], idx[i + 1]] for i in range(len(idx) - 1)))


def chain_mi_score(
    order: tp.List[str],
    feature_names: tp.List[str],
    MI: np.ndarray,
) -> float:
    """
    Chain Mutual Information score (higher = better).

    MI_chain = sum_{i=0}^{n-2}  I(order[i] ; order[i+1])
    """
    idx = [feature_names.index(f) for f in order]
    return float(sum(MI[idx[i], idx[i + 1]] for i in range(len(idx) - 1)))


def score_all_permutations(
    ib_finder: IBOrderFinder,
) -> tp.List[tp.Tuple[float, float, tp.List[str]]]:
    """
    Enumerate all n! permutations and return list of
    (chain_ce, chain_mi, order_list) sorted ascending by chain_ce.
    """
    features = list(ib_finder.feature_names_)
    CE = ib_finder.cond_entropy_
    MI = ib_finder.mi_matrix_

    scored = []
    for perm in itertools.permutations(features):
        ce = chain_ce_score(list(perm), features, CE)
        mi = chain_mi_score(list(perm), features, MI)
        scored.append((ce, mi, list(perm)))

    scored.sort(key=lambda x: x[0])
    return scored


def select_target_orderings(
    ib_finder: IBOrderFinder,
    n_select: int = 5,
    exclude: tp.Optional[tp.List[tp.List[str]]] = None,
) -> tp.List[tp.Tuple[float, float, tp.List[str]]]:
    """
    Select `n_select` orderings evenly spaced across the chain-CE distribution,
    after excluding orderings in `exclude`.

    Returns list of (chain_ce, chain_mi, order) tuples ordered from best to
    worst CE score (p_low to p_high).
    """
    exclude_set = {tuple(o) for o in (exclude or [])}
    all_scored  = score_all_permutations(ib_finder)
    filtered    = [(ce, mi, o) for ce, mi, o in all_scored
                   if tuple(o) not in exclude_set]

    n = len(filtered)
    logger.info(
        f"Permutation pool after exclusion: {n}  "
        f"(total {len(all_scored)}, excluded {len(all_scored) - n})"
    )

    if n == 0:
        raise ValueError(
            "No orderings available after exclusion. "
            "Reduce the exclude list or decrease n_select."
        )

    # Clamp n_select to available pool and warn if needed
    actual_select = min(n_select, n)
    if actual_select < n_select:
        logger.warning(
            f"Only {n} orderings available; reducing n_select "
            f"from {n_select} to {actual_select}."
        )

    step = n / actual_select
    # Pick approximately p(0.5/actual_select), p(1.5/actual_select), ...
    raw_indices = [int(step * (i + 0.5)) for i in range(actual_select)]
    # Clamp to valid range and deduplicate while preserving order
    seen: tp.Set[int] = set()
    indices: tp.List[int] = []
    for idx in raw_indices:
        idx = min(idx, n - 1)
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)

    selected = [filtered[i] for i in indices]

    logger.info(
        f"Chain-CE range: [{filtered[0][0]:.4f}, {filtered[-1][0]:.4f}]"
    )
    for i, (ce, mi, order) in enumerate(selected):
        pct = int(100 * indices[i] / n)
        logger.info(
            f"  ord{i + 1} (p{pct:02d}): CE={ce:.4f}  MI={mi:.4f}  {order}"
        )
    return selected


# ==============================================================================
# Data loading
# ==============================================================================

def load_train_data(subsample: int = 0) -> pd.DataFrame:
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Training CSV not found: {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)
    df["lat"] = df["lat"].round(2)
    df["lon"] = df["lon"].round(2)
    if subsample > 0 and len(df) > subsample:
        df = df.sample(n=subsample, random_state=42).reset_index(drop=True)
    logger.info(f"Train set: {len(df)} rows  cols={list(df.columns)}")
    return df


# ==============================================================================
# Evaluation  (mirrors experiment_order_comparison.py)
# ==============================================================================

def evaluate(syn_df: pd.DataFrame, real_df: pd.DataFrame) -> dict:
    metrics: dict = {}

    for col in CAT_COLS:
        if col not in syn_df.columns:
            metrics[f"{col}_coverage"] = None
            continue
        real_vals = set(real_df[col].dropna().unique())
        cov = float(syn_df[col].dropna().isin(real_vals).mean())
        metrics[f"{col}_coverage"] = round(cov, 4)
        logger.info(f"  [{col}] coverage = {cov:.4f}")

    for col in NUM_COLS:
        if col not in syn_df.columns:
            metrics[f"{col}_mean_err"] = None
            metrics[f"{col}_std_err"]  = None
            continue
        rs = real_df[col].dropna()
        ss = pd.to_numeric(syn_df[col], errors="coerce").dropna()
        if len(ss) == 0:
            metrics[f"{col}_mean_err"] = None
            metrics[f"{col}_std_err"]  = None
            continue
        mean_err = abs(ss.mean() - rs.mean()) / (abs(rs.mean()) + 1e-8)
        std_err  = abs(ss.std()  - rs.std())  / (abs(rs.std())  + 1e-8)
        metrics[f"{col}_mean_err"] = round(float(mean_err), 6)
        metrics[f"{col}_std_err"]  = round(float(std_err),  6)
        logger.info(
            f"  [{col}] real {rs.mean():.2f}+/-{rs.std():.2f}  "
            f"syn {ss.mean():.2f}+/-{ss.std():.2f}  "
            f"mean_err={mean_err:.4f}"
        )
    return metrics


def aggregate_quality(r: dict) -> tp.Optional[float]:
    """
    Scalar quality score in [0, 1] (higher = better).
      quality = 0.6 * avg_coverage + 0.4 * (1 - min(avg_mean_err, 1.0))

    Weights: coverage matters more than numeric precision for the
    ordering-quality question.
    """
    coverages = [
        r.get("state_code_coverage"),
        r.get("bird_coverage"),
        r.get("lat_zone_coverage"),
    ]
    errors = [
        r.get("lat_mean_err"),
        r.get("lon_mean_err"),
    ]
    coverages = [v for v in coverages if v is not None]
    errors    = [v for v in errors    if v is not None]

    if not coverages:
        return None

    avg_cov = float(np.mean(coverages))
    avg_err = float(np.mean(errors)) if errors else 1.0
    return round(0.6 * avg_cov + 0.4 * (1.0 - min(avg_err, 1.0)), 6)


# ==============================================================================
# Scatter plot helper
# ==============================================================================

def run_scatter(synth_csv: str, output_dir: str) -> None:
    script = TIDE_ROOT / "plot_scatter_synthetic.py"
    if not script.exists():
        logger.warning("plot_scatter_synthetic.py not found, skipping")
        return
    cmd = [
        sys.executable, str(script),
        "--synth-csv",  synth_csv,
        "--output-dir", output_dir,
        "--mode",       "both",
        "--n-top",      "6",
        "--n-cols",     "3",
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300)
    except Exception as e:
        logger.warning(f"Scatter plot failed: {e}")


# ==============================================================================
# Model building
# ==============================================================================

def build_model(train_order: tp.List[str], output_dir: str, args) -> FixedOrderGReaT:
    """
    Build a FixedOrderGReaT with the same hyper-parameters as the cond experiment.
    `train_order[0]` will be used as the generation starting column.
    """
    lora_cfg = {
        "r":            args.lora_r,
        "lora_alpha":   args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    model = FixedOrderGReaT(
        fixed_order=train_order,
        llm=args.llm,
        experiment_dir=os.path.join(output_dir, "trainer"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        efficient_finetuning="lora",
        lora_config=lora_cfg,
    )
    if args.bf16:
        model.train_hyperparameters["bf16"] = True
    # Disable checkpoint saving to conserve disk space
    model.train_hyperparameters["save_strategy"] = "no"
    return model


# ==============================================================================
# Single condition runner
# ==============================================================================

def run_condition(
    cond_id:  str,
    order:    tp.List[str],
    ce_score: float,
    mi_score: float,
    df_train: pd.DataFrame,
    args,
) -> dict:
    logger.info("=" * 70)
    logger.info(f"Condition {cond_id}")
    logger.info(f"  order     : {order}")
    logger.info(f"  chain_CE  : {ce_score:.4f}   chain_MI : {mi_score:.4f}")
    logger.info(f"  gen_start : {order[0]}")
    logger.info("=" * 70)

    output_dir = os.path.join(args.base_output_dir, f"cond_{cond_id}")
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # ── Train ──────────────────────────────────────────────────────────────────
    model = build_model(order, output_dir, args)
    model.fit(df_train)
    # After fit(), model.conditional_col == order[0]  (set by FixedOrderGReaT)
    logger.info(f"  conditional_col after fit: {model.conditional_col}")

    # ── Sample ─────────────────────────────────────────────────────────────────
    n_samples = args.n_samples if args.n_samples > 0 else len(df_train)
    syn_df = model.sample(
        n_samples=n_samples,
        guided_sampling=False,
        temperature=args.temperature,
        k=args.sample_k,
        max_length=args.max_length,
    )
    logger.info(f"  Sampled: {len(syn_df)} rows")

    synth_csv = os.path.join(output_dir, "synthetic_us_location.csv")
    syn_df.to_csv(synth_csv, index=False, encoding="utf-8")

    # ── Delete model weights ───────────────────────────────────────────────────
    trainer_dir = os.path.join(output_dir, "trainer")
    if os.path.isdir(trainer_dir):
        shutil.rmtree(trainer_dir)
        logger.info(f"  Weights deleted: {trainer_dir}")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    metrics = evaluate(syn_df, df_train)
    elapsed = time.time() - t0

    summary = {
        "condition":        cond_id,
        "order":            order,
        "chain_ce_score":   round(ce_score, 6),
        "chain_mi_score":   round(mi_score, 6),
        "n_synthetic_rows": int(len(syn_df)),
        "elapsed_seconds":  round(elapsed, 1),
        **metrics,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    run_scatter(synth_csv, output_dir)

    logger.info(
        f"Condition {cond_id} done  {elapsed / 60:.1f} min  "
        f"quality={aggregate_quality(summary)}"
    )
    return summary


# ==============================================================================
# Load existing results from the previous cond experiment
# ==============================================================================

def load_existing_results(
    ib_finder: IBOrderFinder,
) -> tp.Dict[str, tp.Optional[dict]]:
    """
    Try to load summary.json for cond1a, cond1b, cond4, cond5 from
    output_order_comparison/.  Annotates each with chain_ce/mi scores.
    """
    features = list(ib_finder.feature_names_)
    CE = ib_finder.cond_entropy_
    MI = ib_finder.mi_matrix_

    ref_orders = {
        "1a": None,          # IB order, computed below after loading
        "1b": PAFT_ORDER,
        "4":  None,          # random training — no fixed order
        "5":  PAFT_REVERSED,
    }

    results: tp.Dict[str, tp.Optional[dict]] = {}
    for label, ref_order in ref_orders.items():
        path = PREV_OUTPUT_DIR / f"cond_{label}" / "summary.json"
        if not path.exists():
            logger.info(f"  cond{label}: not found at {path}")
            results[label] = None
            continue
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            logger.warning(f"  cond{label}: failed to load — {e}")
            results[label] = None
            continue

        # Annotate with chain_CE/MI scores
        if ref_order is not None:
            d["chain_ce_score"] = round(chain_ce_score(ref_order, features, CE), 6)
            d["chain_mi_score"] = round(chain_mi_score(ref_order, features, MI), 6)
        else:
            # cond1a: derive order from stored train_order field
            stored_order = d.get("train_order")
            if stored_order and isinstance(stored_order, list) and len(stored_order) == len(features):
                d["chain_ce_score"] = round(chain_ce_score(stored_order, features, CE), 6)
                d["chain_mi_score"] = round(chain_mi_score(stored_order, features, MI), 6)
            else:
                d["chain_ce_score"] = None
                d["chain_mi_score"] = None

        results[label] = d
        logger.info(
            f"  cond{label}: loaded  "
            f"CE={d.get('chain_ce_score')}  "
            f"quality={aggregate_quality(d)}"
        )

    return results


# ==============================================================================
# Result visualisation
# ==============================================================================

def plot_results(
    new_results: tp.List[dict],
    existing:    tp.Dict[str, tp.Optional[dict]],
    output_dir:  str,
) -> None:
    """
    Two-panel figure:
      Left:  Scatter of chain_CE score vs aggregate quality.
             Regression line + Pearson r shown.  Tests H1.
      Right: Bar chart comparing all conditions by quality.
             Highlights whether fixed-order conditions cluster together (H2).
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── Panel 1: chain_CE vs quality (scatter) ─────────────────────────────────
    ax = axes[0]

    new_ce  = [r["chain_ce_score"]            for r in new_results]
    new_q   = [aggregate_quality(r)           for r in new_results]
    new_ids = [r["condition"]                 for r in new_results]

    # New conditions (blue)
    valid_new = [(ce, q, cid)
                 for ce, q, cid in zip(new_ce, new_q, new_ids)
                 if q is not None and ce is not None]
    if valid_new:
        ces_n, qs_n, ids_n = zip(*valid_new)
        ax.scatter(ces_n, qs_n, color="steelblue", s=120, zorder=4,
                   label="New (exp1)")
        for ce, q, cid in zip(ces_n, qs_n, ids_n):
            ax.annotate(cid, (ce, q), textcoords="offset points",
                        xytext=(5, 3), fontsize=8)

    # Existing reference conditions
    ref_style = {
        "1a": ("green",  "^", "cond1a (IB opt)"),
        "1b": ("green",  "s", "cond1b (PAFT opt)"),
        "4":  ("gray",   "o", "cond4 (baseline)"),
        "5":  ("orange", "D", "cond5 (PAFT rev)"),
    }
    for k, d in existing.items():
        if d is None:
            continue
        ce = d.get("chain_ce_score")
        q  = aggregate_quality(d)
        if ce is None or q is None:
            continue
        color, marker, label = ref_style.get(k, ("red", "o", f"cond{k}"))
        ax.scatter([ce], [q], color=color, marker=marker, s=160,
                   zorder=5, label=label)
        ax.annotate(f"cond{k}", (ce, q), textcoords="offset points",
                    xytext=(5, 3), fontsize=8)

    # Regression line on ALL points with valid CE scores
    all_ce = [r["chain_ce_score"] for r in new_results if r.get("chain_ce_score") is not None]
    all_q  = [aggregate_quality(r) for r in new_results if aggregate_quality(r) is not None]
    for d in existing.values():
        if d and d.get("chain_ce_score") is not None and aggregate_quality(d) is not None:
            all_ce.append(d["chain_ce_score"])
            all_q.append(aggregate_quality(d))

    if len(all_ce) >= 3:
        z = np.polyfit(all_ce, all_q, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(all_ce), max(all_ce), 200)
        ax.plot(x_line, p(x_line), "k--", alpha=0.4, linewidth=1.2,
                label="regression")
        corr = np.corrcoef(all_ce, all_q)[0, 1]
        ax.set_title(
            f"Chain CE Score vs Generation Quality\n"
            f"Pearson r = {corr:.3f}  "
            f"({'H1 supported' if abs(corr) > 0.5 else 'H1 not supported'})",
            fontsize=10,
        )
    else:
        ax.set_title("Chain CE Score vs Generation Quality", fontsize=10)

    ax.set_xlabel("Chain CE Score (lower = better ordering)", fontsize=9)
    ax.set_ylabel("Aggregate Quality (higher = better)", fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: bar chart of quality by condition ─────────────────────────────
    ax2 = axes[1]

    bar_data: tp.List[tp.Tuple[str, tp.Optional[float], str]] = []
    for r in new_results:
        bar_data.append((r["condition"], aggregate_quality(r), "steelblue"))
    for k, d in existing.items():
        if d is not None:
            color = {"1a": "green", "1b": "limegreen",
                     "4": "gray", "5": "orange"}.get(k, "red")
            bar_data.append((f"cond{k}", aggregate_quality(d), color))

    bar_data = [(lbl, q, c) for lbl, q, c in bar_data if q is not None]
    bar_data.sort(key=lambda x: x[1], reverse=True)

    labels = [x[0] for x in bar_data]
    values = [x[1] for x in bar_data]
    colors = [x[2] for x in bar_data]

    bars = ax2.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax2.set_title("Quality Comparison: All Conditions", fontsize=10)
    ax2.set_ylabel("Aggregate Quality (0.6×coverage + 0.4×(1-err))", fontsize=9)
    ax2.set_xlabel("Condition", fontsize=9)
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_ylim(0, min(1.05, max(values) * 1.15) if values else 1.05)

    # Legend for bar chart
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="steelblue", label="Exp1 (new orderings)"),
        Patch(facecolor="green",     label="cond1a (IB opt)"),
        Patch(facecolor="limegreen", label="cond1b (PAFT opt)"),
        Patch(facecolor="gray",      label="cond4 (random baseline)"),
        Patch(facecolor="orange",    label="cond5 (PAFT rev)"),
    ]
    ax2.legend(handles=legend_elems, fontsize=7, loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "results_summary.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Summary plot saved: {out_path}")


def save_score_distribution(
    all_scored: tp.List[tp.Tuple[float, float, tp.List[str]]],
    selected:   tp.List[tp.Tuple[float, float, tp.List[str]]],
    output_dir: str,
) -> None:
    """Save a histogram of all 120 chain-CE scores with selected points marked."""
    all_ces = [ce for ce, _, _ in all_scored]
    sel_ces = [ce for ce, _, _ in selected]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_ces, bins=20, color="lightsteelblue", edgecolor="white",
            label="All 120 permutations")
    for ce in sel_ces:
        ax.axvline(ce, color="steelblue", linestyle="--", linewidth=1.5)
    # Reference lines
    paft_ce = [ce for ce, _, o in all_scored if o == PAFT_ORDER]
    rev_ce  = [ce for ce, _, o in all_scored if o == PAFT_REVERSED]
    if paft_ce:
        ax.axvline(paft_ce[0], color="green",  linestyle="-", linewidth=2,
                   label=f"PAFT_ORDER (CE={paft_ce[0]:.3f})")
    if rev_ce:
        ax.axvline(rev_ce[0],  color="orange", linestyle="-", linewidth=2,
                   label=f"PAFT_REVERSED (CE={rev_ce[0]:.3f})")

    ax.set_xlabel("Chain CE Score (lower = better)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Distribution of Chain CE Scores (all 5! = 120 orderings)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, "score_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Score distribution plot saved: {out_path}")


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    args = parse_args()
    os.makedirs(args.base_output_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    df_train = load_train_data(args.subsample)

    # ── Fit IBOrderFinder ─────────────────────────────────────────────────────
    logger.info("Fitting IBOrderFinder...")
    ib_finder = IBOrderFinder(n_bins=100, mi_threshold_quantile=0.25)
    ib_finder.fit(df_train)
    IB_ORDER = ib_finder.get_order()
    logger.info(f"IB order   : {IB_ORDER}")
    logger.info(f"PAFT order : {PAFT_ORDER}")

    # ── Score reference orderings ─────────────────────────────────────────────
    features = list(ib_finder.feature_names_)
    CE = ib_finder.cond_entropy_
    MI = ib_finder.mi_matrix_

    all_scored = score_all_permutations(ib_finder)

    logger.info("Reference ordering scores:")
    for name, order in [("IB_ORDER", IB_ORDER),
                        ("PAFT_ORDER", PAFT_ORDER),
                        ("PAFT_REVERSED", PAFT_REVERSED)]:
        ce = chain_ce_score(order, features, CE)
        mi = chain_mi_score(order, features, MI)
        all_ces = [s for s, _, _ in all_scored]
        pct = int(100 * sum(s <= ce for s in all_ces) / len(all_ces))
        logger.info(f"  {name}: CE={ce:.4f}  MI={mi:.4f}  percentile={pct}")

    # ── Select 5 new orderings ─────────────────────────────────────────────────
    exclude = [PAFT_ORDER, PAFT_REVERSED, IB_ORDER]
    selected = select_target_orderings(
        ib_finder, n_select=args.n_select, exclude=exclude
    )

    # ── Save score distribution plot ──────────────────────────────────────────
    save_score_distribution(all_scored, selected, args.base_output_dir)

    # ── Save ordering metadata ─────────────────────────────────────────────────
    ordering_metadata = {
        "timestamp":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "PAFT_ORDER":   {
            "order":    PAFT_ORDER,
            "chain_ce": round(chain_ce_score(PAFT_ORDER, features, CE), 6),
            "chain_mi": round(chain_mi_score(PAFT_ORDER, features, MI), 6),
        },
        "PAFT_REVERSED": {
            "order":    PAFT_REVERSED,
            "chain_ce": round(chain_ce_score(PAFT_REVERSED, features, CE), 6),
            "chain_mi": round(chain_mi_score(PAFT_REVERSED, features, MI), 6),
        },
        "IB_ORDER": {
            "order":    IB_ORDER,
            "chain_ce": round(chain_ce_score(IB_ORDER, features, CE), 6),
            "chain_mi": round(chain_mi_score(IB_ORDER, features, MI), 6),
        },
        "selected_orderings": [
            {
                "condition": f"ord{i + 1}",
                "order":     o,
                "chain_ce":  round(ce, 6),
                "chain_mi":  round(mi, 6),
            }
            for i, (ce, mi, o) in enumerate(selected)
        ],
    }
    with open(
        os.path.join(args.base_output_dir, "orderings.json"),
        "w", encoding="utf-8",
    ) as f:
        json.dump(ordering_metadata, f, ensure_ascii=False, indent=2)

    # ── Build condition list ──────────────────────────────────────────────────
    all_conditions = [
        (f"ord{i + 1}", ce, mi, o)
        for i, (ce, mi, o) in enumerate(selected)
    ]
    if args.conditions:
        all_conditions = [c for c in all_conditions if c[0] in args.conditions]
        if not all_conditions:
            logger.error(
                f"No matching conditions for --conditions {args.conditions}. "
                f"Available: {[f'ord{i+1}' for i in range(args.n_select)]}"
            )
            return

    # ── Run conditions ─────────────────────────────────────────────────────────
    all_results: tp.List[dict] = []
    for cond_id, ce, mi, order in all_conditions:
        try:
            result = run_condition(cond_id, order, ce, mi, df_train, args)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Condition {cond_id} failed: {e}", exc_info=True)

    if not all_results:
        logger.error("No conditions completed successfully.")
        return

    # ── Load existing cond experiment results ─────────────────────────────────
    logger.info("Loading existing condition results...")
    existing = load_existing_results(ib_finder)

    # ── Print summary table ───────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info(
        f"{'Cond':<8} {'CE':>8} {'MI':>8} "
        f"{'state_cov':>10} {'bird_cov':>9} {'lat_err':>9} {'lon_err':>9} "
        f"{'quality':>9}"
    )
    for r in all_results:
        q = aggregate_quality(r)
        logger.info(
            f"{r['condition']:<8} "
            f"{r['chain_ce_score']:>8.4f} "
            f"{r['chain_mi_score']:>8.4f} "
            f"{str(r.get('state_code_coverage', '')):>10} "
            f"{str(r.get('bird_coverage', '')):>9} "
            f"{str(r.get('lat_mean_err', '')):>9} "
            f"{str(r.get('lon_mean_err', '')):>9} "
            f"{(f'{q:.4f}' if q is not None else 'N/A'):>9}"
        )
    logger.info("=" * 80)

    # ── Save all results ───────────────────────────────────────────────────────
    with open(
        os.path.join(args.base_output_dir, "all_results.json"),
        "w", encoding="utf-8",
    ) as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(all_results, existing, args.base_output_dir)

    logger.info(f"All outputs written to: {args.base_output_dir}")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exp1: Order Quality vs Training-Generation Consistency",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Output
    p.add_argument(
        "--base_output_dir",
        default=str(HERE / "output"),
        help="Directory for all outputs",
    )
    # Model / training  (match experiment_order_comparison.py defaults)
    p.add_argument("--llm",          default="gpt2-medium")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lora_r",       type=int,   default=8)
    p.add_argument("--lora_alpha",   type=int,   default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--bf16",         action="store_true", default=True)
    p.add_argument("--no-bf16",      dest="bf16", action="store_false")
    # Sampling
    p.add_argument("--temperature",  type=float, default=0.8)
    p.add_argument("--max_length",   type=int,   default=100)
    p.add_argument("--sample_k",     type=int,   default=100)
    p.add_argument("--n_samples",    type=int,   default=0,
                   help="Generated rows; 0 = same as training set")
    # Data
    p.add_argument("--subsample",    type=int,   default=0,
                   help="Sub-sample training set; 0 = full")
    # Experiment control
    p.add_argument("--n_select",     type=int,   default=5,
                   help="Number of new orderings to select (default 5)")
    p.add_argument("--conditions",   nargs="+",  default=None,
                   help="Run only specific IDs, e.g. --conditions ord1 ord3")
    return p.parse_args()


if __name__ == "__main__":
    main()

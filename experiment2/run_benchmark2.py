"""
run_benchmark2.py — Experiment 2: Evaluate Concept-Drift Synthetic Data
========================================================================
Evaluates synthetic CSVs produced by generate_synthetic2.py against their
real counterparts in:

    llm_gtd_benchmark/data/concept_drift_data/origin_data/<dataset>/

Dimensions evaluated:
  Dim0  structural validity  (IRR)
  Dim1  distributional fidelity (KS / TVD / alpha-precision / beta-recall / C2ST)
  Dim2  logical consistency  (DSI; no LogicSpec → ICVR/HCS/MDI skipped)
  Dim3  ML utility           (TSTR — binary classification, target col = 'y')
  Dim4  privacy              (DCR 5th / 95th percentile)
  [Dim5 skipped — no sensitive attributes defined for drift datasets]

Results saved to:
  results2/<model>_<dataset>.json    per-run ResultBundle
  results2/leaderboard.csv           aggregated leaderboard

NOTE: The time-index column ``t`` is dropped from both train and synthetic
      data before evaluation, matching the preprocessing in generate_synthetic2.py.
"""

from __future__ import annotations

import logging
import sys
import traceback
import warnings
from pathlib import Path

import glob as _glob
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from llm_gtd_benchmark import (
    BenchmarkPipeline,
    DataSchema,
    PipelineConfig,
    ResultBundle,
)
from llm_gtd_benchmark.metrics.dimension3 import TaskType

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_benchmark2")

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = ROOT / "llm_gtd_benchmark" / "data"
ORIGIN_ROOT = DATA_ROOT / "concept_drift_data" / "origin_data"
SYNTH_ROOT  = DATA_ROOT / "concept_drift_data" / "synthetic_data"
RESULTS_DIR = ROOT / "results2"
RESULTS_DIR.mkdir(exist_ok=True)

# ── dataset configurations ────────────────────────────────────────────────────
# All concept-drift datasets share the same target column and task type.
# Dim5 (fairness) is skipped globally — no protected attributes.
DATASETS: dict[str, dict] = {

    "agrawal": dict(
        target_col="y",
        task_type=TaskType.BINARY_CLASS,
    ),

    "hyperplane": dict(
        target_col="y",
        task_type=TaskType.BINARY_CLASS,
    ),

    "rbfdrift": dict(
        target_col="y",
        task_type=TaskType.BINARY_CLASS,
    ),

    "sea": dict(
        target_col="y",
        task_type=TaskType.BINARY_CLASS,
    ),

    "stagger": dict(
        target_col="y",
        task_type=TaskType.BINARY_CLASS,
    ),
}

MODELS = ["GReaT", "GraFT", "GraDe", "PAFT"]

# ── helpers ───────────────────────────────────────────────────────────────────

def find_synth(model: str, dataset: str) -> Path:
    pattern = str(SYNTH_ROOT / model / f"*{dataset}*.csv")
    files = _glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No synthetic file found: {pattern}")
    return Path(files[0])


def drop_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the ``t`` column (time index) if present."""
    return df.drop(columns=["t"], errors="ignore")


def align_columns(synth_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure synthetic DataFrame has exactly the same columns as real_df."""
    missing = [c for c in real_df.columns if c not in synth_df.columns]
    if missing:
        synth_df = synth_df.copy()
        for c in missing:
            logger.warning("Column '%s' missing from synthetic data — filling with NaN.", c)
            synth_df[c] = float("nan")
    return synth_df[real_df.columns]


# ── main evaluation loop ──────────────────────────────────────────────────────

def run_all() -> dict:
    bundles: dict = {}

    for dataset_name, ds_cfg in DATASETS.items():
        logger.info("=" * 70)
        logger.info("DATASET: %s", dataset_name.upper())
        logger.info("=" * 70)

        train_df = drop_time_index(pd.read_csv(ORIGIN_ROOT / dataset_name / "train.csv"))
        test_df  = drop_time_index(pd.read_csv(ORIGIN_ROOT / dataset_name / "test.csv"))
        schema   = DataSchema(train_df)
        logger.info("Schema: %s", schema)

        for model_name in MODELS:
            out_path = RESULTS_DIR / f"{model_name}_{dataset_name}.json"
            if out_path.exists():
                logger.info("SKIP %s/%s — result already exists.", model_name, dataset_name)
                try:
                    bundles[(model_name, dataset_name)] = ResultBundle.load(out_path)
                except Exception:
                    pass
                continue

            logger.info("─" * 60)
            logger.info("MODEL: %s  |  DATASET: %s", model_name, dataset_name)

            try:
                synth_path = find_synth(model_name, dataset_name)
            except FileNotFoundError as e:
                logger.error("%s", e)
                continue

            synth_df = drop_time_index(pd.read_csv(synth_path))
            synth_df = align_columns(synth_df, train_df)
            logger.info("Loaded synthetic: %s  shape=%s", synth_path.name, synth_df.shape)

            if len(synth_df) == 0:
                logger.error("Synthetic data is empty for %s/%s — skipping.", model_name, dataset_name)
                continue

            config = PipelineConfig(
                schema=schema,
                train_real_df=train_df,
                test_real_df=test_df,
                model_name=model_name,
                dataset_name=dataset_name,
                target_col=ds_cfg["target_col"],
                task_type=ds_cfg["task_type"],
                fair_spec=None,                    # Dim5 skipped
                dimensions=[0, 1, 2, 3, 4],        # skip Dim5 fairness
                n_boot=0,
                random_state=42,
                notes=f"exp2,synth_file={synth_path.name}",
            )

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    bundle = BenchmarkPipeline(config).run(synth_df)
            except Exception:
                logger.error("Pipeline CRASHED for %s/%s:\n%s",
                             model_name, dataset_name, traceback.format_exc())
                continue

            bundle.save(out_path)
            bundles[(model_name, dataset_name)] = bundle

            logger.info("Dims computed: %s", bundle.dimensions_computed)
            if bundle.errors:
                for dim, msg in bundle.errors.items():
                    lvl = "SKIP" if msg.startswith("skipped:") else "FAIL"
                    logger.warning("  %s %s: %s", lvl, dim, msg[:120])

    return bundles


# ── leaderboard ───────────────────────────────────────────────────────────────

def build_leaderboard(bundles: dict) -> pd.DataFrame:
    rows = []
    for (model, dataset), b in sorted(bundles.items()):
        r0, r1, r2, r3, r4 = b.result0, b.result1, b.result2, b.result3, b.result4

        def g(obj, attr, default=float("nan")):
            if obj is None:
                return default
            v = getattr(obj, attr, default)
            return v if v is not None else default

        def g_dict_mean(obj, attr):
            if obj is None:
                return float("nan")
            d = getattr(obj, attr, None)
            if not d:
                return float("nan")
            vals = []
            for v in d.values():
                if isinstance(v, dict):
                    vals.extend(x for x in v.values() if isinstance(x, float) and x == x)
                elif isinstance(v, float) and v == v:
                    vals.append(v)
            return sum(vals) / len(vals) if vals else float("nan")

        rows.append({
            "model":                 model,
            "dataset":               dataset,
            # Dim 0
            "irr":                   g(r0, "irr"),
            "n_clean":               g(r0, "n_clean", 0),
            # Dim 1
            "mean_ks":               g(r1, "mean_ks"),
            "mean_tvd":              g(r1, "mean_tvd"),
            "alpha_precision":       g(r1, "alpha_precision"),
            "beta_recall":           g(r1, "beta_recall"),
            "c2st_auc_mean":         g(r1, "c2st_auc_mean"),
            # Dim 2
            "dsi_gap":               g(r2, "dsi_gap"),
            "dsi_relative_gap_pct":  g(r2, "dsi_relative_gap"),
            "icvr":                  g(r2, "icvr"),
            # Dim 3
            "mle_tstr_primary":      g_dict_mean(r3, "mle_tstr"),
            # Dim 4
            "dcr_5th":               g(r4, "dcr_5th_percentile"),
            "dcr_95th":              g(r4, "dcr_95th_percentile"),
            "exact_match_rate":      g(r4, "exact_match_rate"),
            # Errors
            "n_errors":              sum(1 for v in b.errors.values() if not v.startswith("skipped:")),
            "skipped_dims":          ", ".join(k for k, v in b.errors.items() if v.startswith("skipped:")),
        })

    return pd.DataFrame(rows)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bundles = run_all()

    if bundles:
        lb = build_leaderboard(bundles)
        csv_path = RESULTS_DIR / "leaderboard.csv"
        lb.to_csv(csv_path, index=False, float_format="%.4f")
        logger.info("Leaderboard saved -> %s", csv_path)

        print("\n" + "=" * 80)
        print("LEADERBOARD — Experiment 2 (Concept Drift)")
        print("=" * 80)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(lb.to_string(index=False))
    else:
        logger.warning("No bundles produced — check that synthetic CSVs exist in %s.", SYNTH_ROOT)

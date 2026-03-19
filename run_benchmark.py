"""
LLM-GTD Benchmark — Full Experiment Runner
==========================================
5 datasets × 4 models = 20 evaluation runs.

Dimensions evaluated:
  Dim0  structural validity (IRR)
  Dim1  distributional fidelity (KS / TVD / alpha-precision / beta-recall / C2ST)
  Dim2  logical consistency (DSI; no LogicSpec → ICVR/HCS/MDI skipped)
  Dim3  ML utility (TSTR)
  Dim4  privacy (DCR)
  Dim5  fairness — income dataset only (protected: race, gender)

Results are saved to:
  results/<model>_<dataset>.json     per-run ResultBundle
  results/leaderboard.csv            aggregated leaderboard
"""

import logging
import os
import sys
import traceback
import warnings
from pathlib import Path

import pandas as pd

# ── project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from llm_gtd_benchmark import (
    BenchmarkPipeline,
    DataSchema,
    PipelineConfig,
    ResultBundle,
)
from llm_gtd_benchmark.metrics.dimension3 import TaskType
from llm_gtd_benchmark.metrics.dimension5 import FairSpec

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_benchmark")

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT = ROOT / "llm_gtd_benchmark" / "data"
ORIGIN_ROOT = DATA_ROOT / "origin_data"
SYNTH_ROOT  = DATA_ROOT / "synthetic"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── dataset configurations ────────────────────────────────────────────────────
DATASETS = {
    "diabetes": dict(
        target_col="Diabetes",
        task_type=TaskType.BINARY_CLASS,
        fair_spec=None,
    ),
    "house": dict(
        target_col="median_house_value",
        task_type=TaskType.REGRESSION,
        fair_spec=None,
    ),
    "income": dict(
        target_col="income",
        task_type=TaskType.BINARY_CLASS,
        fair_spec=FairSpec(
            protected_cols=["race", "gender"],
            target_col="income",
            task_type=TaskType.BINARY_CLASS,
            intersectional=False,
            min_group_size=30,
        ),
    ),
    "sick": dict(
        target_col="Class",
        task_type=TaskType.BINARY_CLASS,
        fair_spec=None,
    ),
    "us_location": dict(
        target_col="lat_zone",
        task_type=TaskType.MULTI_CLASS,
        fair_spec=None,
    ),
}

# ── model → synthetic file patterns ──────────────────────────────────────────
import glob as _glob

def find_synth(model: str, dataset: str) -> Path:
    pattern = str(SYNTH_ROOT / model / f"*{dataset}*.csv")
    files = _glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No synthetic file found: {pattern}")
    return Path(files[0])


def align_columns(synth_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    """Drop extra columns and add any missing columns (as NaN) so schemas match."""
    missing = [col for col in real_df.columns if col not in synth_df.columns]
    if missing:
        synth_df = synth_df.copy()
        for col in missing:
            logger.warning("Column '%s' missing from synthetic data — filling with NaN.", col)
            synth_df[col] = float("nan")
    return synth_df[real_df.columns]


# ── main loop ─────────────────────────────────────────────────────────────────
MODELS = ["GreaT", "GraFT", "GraDe", "PAFT"]

def run_all():
    bundles: dict = {}  # (model, dataset) -> ResultBundle

    for dataset_name, ds_cfg in DATASETS.items():
        logger.info("=" * 70)
        logger.info("DATASET: %s", dataset_name.upper())
        logger.info("=" * 70)

        train_df = pd.read_csv(ORIGIN_ROOT / dataset_name / "train.csv")
        test_df  = pd.read_csv(ORIGIN_ROOT / dataset_name / "test.csv")

        schema = DataSchema(train_df)
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

            synth_df = pd.read_csv(synth_path)
            synth_df = align_columns(synth_df, train_df)
            logger.info("Loaded synthetic: %s  shape=%s", synth_path.name, synth_df.shape)

            config = PipelineConfig(
                schema=schema,
                train_real_df=train_df,
                test_real_df=test_df,
                model_name=model_name,
                dataset_name=dataset_name,
                target_col=ds_cfg["target_col"],
                task_type=ds_cfg["task_type"],
                fair_spec=ds_cfg["fair_spec"],
                dimensions=[0, 1, 2, 3, 4],   # skip Dim5 fairness
                n_boot=0,          # disable bootstrap for speed; enable with n_boot=500
                random_state=42,
                notes=f"synth_file={synth_path.name}",
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

            logger.info("Dims computed : %s", bundle.dimensions_computed)
            if bundle.errors:
                for dim, msg in bundle.errors.items():
                    lvl = "SKIP" if msg.startswith("skipped:") else "FAIL"
                    logger.warning("  %s %s: %s", lvl, dim, msg[:120])

    return bundles


# ── leaderboard ───────────────────────────────────────────────────────────────
def build_leaderboard(bundles: dict) -> pd.DataFrame:
    rows = []
    for (model, dataset), b in sorted(bundles.items()):
        r0 = b.result0
        r1 = b.result1
        r2 = b.result2
        r3 = b.result3
        r4 = b.result4
        r5 = b.result5

        def g(obj, attr, default=float("nan")):
            if obj is None:
                return default
            v = getattr(obj, attr, default)
            return v if v is not None else default

        def g_dict_mean(obj, attr):
            """Mean of all finite floats in a flat or one-level-nested dict.

            Handles both:
              - flat:   {key: float}
              - nested: {key: {metric: float}}   ← mle_tstr / utility_gap shape
            """
            if obj is None:
                return float("nan")
            d = getattr(obj, attr, None)
            if not d:
                return float("nan")
            vals: list = []
            for v in d.values():
                if isinstance(v, dict):
                    # nested: {metric: float}
                    vals.extend(
                        x for x in v.values() if isinstance(x, float) and x == x
                    )
                elif isinstance(v, float) and v == v:
                    vals.append(v)
            return sum(vals) / len(vals) if vals else float("nan")

        row = {
            "model":          model,
            "dataset":        dataset,
            # Dim 0
            "irr":            g(r0, "irr"),
            "n_clean":        g(r0, "n_clean", 0),
            # Dim 1
            "mean_ks":        g(r1, "mean_ks"),
            "mean_tvd":       g(r1, "mean_tvd"),
            "alpha_precision":g(r1, "alpha_precision"),
            "beta_recall":    g(r1, "beta_recall"),
            "c2st_auc_mean":  g(r1, "c2st_auc_mean"),
            # Dim 2
            "dsi_gap":        g(r2, "dsi_gap"),
            "dsi_relative_gap_pct": g(r2, "dsi_relative_gap"),
            "icvr":           g(r2, "icvr"),
            # Dim 3
            "mle_tstr_primary": g_dict_mean(r3, "mle_tstr"),
            # Dim 4
            "dcr_5th":        g(r4, "dcr_5th_percentile"),
            "dcr_95th":       g(r4, "dcr_95th_percentile"),
            "exact_match_rate":g(r4, "exact_match_rate"),
            # Dim 5
            "delta_dp_mean":  g_dict_mean(r5, "delta_dp"),
            "delta_eo_mean":  g_dict_mean(r5, "delta_eo"),
            # Errors
            "n_errors":       sum(1 for v in b.errors.values() if not v.startswith("skipped:")),
            "skipped_dims":   ", ".join(k for k, v in b.errors.items() if v.startswith("skipped:")),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bundles = run_all()

    if bundles:
        lb = build_leaderboard(bundles)
        csv_path = RESULTS_DIR / "leaderboard.csv"
        lb.to_csv(csv_path, index=False, float_format="%.4f")
        logger.info("Leaderboard saved -> %s", csv_path)

        # Print summary
        print("\n" + "=" * 80)
        print("LEADERBOARD")
        print("=" * 80)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(lb.to_string(index=False))
    else:
        logger.warning("No bundles produced.")

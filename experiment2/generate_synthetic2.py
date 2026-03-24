"""
generate_synthetic2.py — Experiment 2: Concept-Drift Datasets
=============================================================
Trains each of the four generative models (GReaT, PAFT, GraDe, GraFT) on
concept-drift benchmark datasets and writes synthetic samples to:

    llm_gtd_benchmark/data/concept_drift_data/synthetic_data/<ModelName>/<dataset>_synth.csv

The time-index column ``t`` is dropped before training; it is a sequential
row counter that carries no generative information.

After generation, run ``python run_benchmark2.py`` to evaluate all files.

Usage
-----
# All models × all active datasets
python generate_synthetic2.py

# Specific model / dataset
python generate_synthetic2.py --models GReaT PAFT --datasets agrawal sea

# Dry-run: print what would run without training
python generate_synthetic2.py --dry-run

# Force re-generation even if output CSV exists
python generate_synthetic2.py --force
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
import warnings

# Suppress PyTorch pin_memory DeprecationWarning from DataLoader workers.
os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore::DeprecationWarning:torch.utils.data._utils.pin_memory",
)
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*is_pinned.*",  category=DeprecationWarning)

from datetime import datetime
from pathlib import Path

import pandas as pd

# ── project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from llm_gtd_benchmark.models import GReaTModel, PAFTModel, GraDeModel, GraFTModel
from llm_gtd_benchmark.models.paft import discover_fd_order

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_synthetic2")

# ── experiment log paths ──────────────────────────────────────────────────────
EXP_LOG_PATH = ROOT / "experiment2_log.jsonl"
EXP_CSV_PATH = ROOT / "experiment2_log.csv"

_EXP_RECORDS: list[dict] = []

def _append_exp_log(record: dict) -> None:
    _EXP_RECORDS.append(record)
    with EXP_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def _flush_exp_csv() -> None:
    pd.DataFrame(_EXP_RECORDS).to_csv(EXP_CSV_PATH, index=False, float_format="%.4f")

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = ROOT / "llm_gtd_benchmark" / "data"
ORIGIN_ROOT = DATA_ROOT / "concept_drift_data" / "origin_data"
SYNTH_ROOT  = DATA_ROOT / "concept_drift_data" / "synthetic_data"
CKPT_BASE   = ROOT / "checkpoints2"

# ── LLM / training hyper-parameters ──────────────────────────────────────────
LLM           = "distilgpt2"
LEARNING_RATE = 5e-4
TOP_P         = 1.0
SEED          = 42

# LoRA
LORA_R              = 8
LORA_ALPHA          = 16
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["c_attn", "c_proj"]

# ── per-dataset configs ────────────────────────────────────────────────────────
#
#  epochs      : fine-tuning epochs
#  batch_size  : per-device training batch size
#  max_length  : max token length per generated row
#  sample_k    : top-k for legacy sampling
#  train_order : GraDe/GraFT fixed column order (None = auto-discover via IB)
#  fd_list     : GraDe/GraFT FD constraints ([] = none known)
#  drop_cols   : columns dropped before training (t is handled globally)
#
# NOTE: ``t`` (time index) is ALWAYS dropped globally in generate_one().
#       Add any additional dataset-specific columns to drop_cols.
#
DATASET_CONFIGS: dict[str, dict] = {

    # ── Active datasets ────────────────────────────────────────────────────────

    "agrawal": {
        # 10 features: salary, commission, age, elevel, car, zipcode,
        #              hvalue, hyears, loan, y  (after dropping t)
        # IB order not pre-computed; GraDe/GraFT use auto-discovery.
        "train_order": None,
        "fd_list":     [],
        "start_col":   None,
        "epochs":      100,
        "batch_size":  16,
        "max_length":  200,
        "sample_k":    50,
        "drop_cols":   [],
    },

    "hyperplane": {
        # 10 anonymous float features + binary y
        "train_order": None,
        "fd_list":     [],
        "start_col":   None,
        "epochs":      100,
        "batch_size":  16,
        "max_length":  200,
        "sample_k":    50,
        "drop_cols":   [],
    },

    "rbfdrift": {
        # 10 anonymous float features + binary y
        "train_order": None,
        "fd_list":     [],
        "start_col":   None,
        "epochs":      100,
        "batch_size":  16,
        "max_length":  200,
        "sample_k":    50,
        "drop_cols":   [],
    },

    "sea": {
        # 3 float features + binary y  — simple, fast
        "train_order": None,
        "fd_list":     [],
        "start_col":   None,
        "epochs":      100,
        "batch_size":  32,
        "max_length":  100,
        "sample_k":    100,
        "drop_cols":   [],
    },

    "stagger": {
        # 3 categorical features (size, color, shape) + binary y
        "train_order": None,
        "fd_list":     [],
        "start_col":   None,
        "epochs":      100,
        "batch_size":  32,
        "max_length":  100,
        "sample_k":    100,
        "drop_cols":   [],
    },
}

ALL_DATASETS = list(DATASET_CONFIGS.keys())
ALL_MODELS   = ["GReaT", "PAFT", "GraDe", "GraFT"]

# ── GPU helpers ───────────────────────────────────────────────────────────────

def _reset_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

def _gpu_snapshot() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "gpu_peak_gb":  round(torch.cuda.max_memory_allocated() / 1e9, 3),
                "gpu_alloc_gb": round(torch.cuda.memory_allocated()     / 1e9, 3),
                "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 3),
            }
    except Exception:
        pass
    return {}

def _free_gpu() -> None:
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

# ── checkpoint helpers ────────────────────────────────────────────────────────

def _model_save_dir(ckpt_base: Path, model_name: str, dataset_name: str) -> Path:
    return ckpt_base / model_name / f"{dataset_name}_saved"

def _meta_path(save_dir: Path) -> Path:
    return save_dir / ".meta.json"

def _save_model(model, model_name: str, save_dir: Path,
                fitted_order, train_time_s) -> None:
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        if model_name in ("GReaT", "PAFT"):
            model.save(str(save_dir))
        elif model_name in ("GraDe", "GraFT"):
            model.save(str(save_dir))
        meta = {"fitted_order": fitted_order, "train_time_s": train_time_s}
        _meta_path(save_dir).write_text(json.dumps(meta), encoding="utf-8")
        logger.info("  Checkpoint saved  →  %s", save_dir)
    except Exception:
        logger.warning("  Checkpoint save failed:\n%s", traceback.format_exc())

def _load_saved_model(model_name: str, save_dir: Path, device: str):
    meta = json.loads(_meta_path(save_dir).read_text(encoding="utf-8"))
    fitted_order = meta.get("fitted_order")
    train_time_s = meta.get("train_time_s")
    if model_name == "GReaT":
        model = GReaTModel.load(str(save_dir), device=device)
    elif model_name == "PAFT":
        model = PAFTModel.load(str(save_dir), device=device)
    elif model_name == "GraDe":
        model = GraDeModel.load(str(save_dir), device=device)
    elif model_name == "GraFT":
        model = GraFTModel.load(str(save_dir), device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model, fitted_order, train_time_s

# ── model factory ─────────────────────────────────────────────────────────────

def build_model(
    model_name: str,
    train_df: pd.DataFrame,
    dataset_name: str,
    device: str,
    paft_use_hyfd: bool,
    ckpt_base: Path,
):
    cfg      = DATASET_CONFIGS[dataset_name]
    ckpt_dir = str(ckpt_base / model_name / dataset_name)

    try:
        import torch
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        use_bf16 = False

    common_train_kwargs = dict(
        learning_rate=LEARNING_RATE,
        save_steps=500,
        logging_steps=100,
        bf16=use_bf16,
        fp16=False,
    )

    if model_name == "GReaT":
        return GReaTModel(
            llm=LLM,
            experiment_dir=ckpt_dir,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            use_lora=True,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            **common_train_kwargs,
        )

    elif model_name == "PAFT":
        column_order = None
        if paft_use_hyfd:
            logger.info("  [PAFT] Running HyFD to discover FD column order ...")
            try:
                column_order = discover_fd_order(train_df)
                logger.info("  [PAFT] HyFD order: %s", column_order)
            except Exception as exc:
                logger.warning(
                    "  [PAFT] HyFD failed (%s) — using random permutation.", exc
                )
        return PAFTModel(
            llm=LLM,
            experiment_dir=ckpt_dir,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            column_order=column_order,
            random_state=SEED,
            use_lora=True,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            **common_train_kwargs,
        )

    elif model_name == "GraDe":
        fd_list = cfg.get("fd_list") or []
        return GraDeModel(
            llm=LLM,
            experiment_dir=ckpt_dir,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            sparsity_lambda=0.001,
            use_dynamic_graph=True,
            num_head_groups=4,
            fd_lambda=0.1,
            fd_alpha=0.5,
            fd_list=fd_list if fd_list else None,
            device=device,
            use_lora=True,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            lora_target_modules=LORA_TARGET_MODULES,
            **common_train_kwargs,
        )

    elif model_name == "GraFT":
        fd_list = cfg.get("fd_list") or []
        return GraFTModel(
            llm=LLM,
            experiment_dir=ckpt_dir,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            n_bins=100,
            mi_threshold_quantile=0.25,
            sparsity_lambda=0.001,
            use_dynamic_graph=True,
            num_head_groups=4,
            fd_lambda=0.1,
            fd_alpha=0.5,
            fd_list=fd_list if fd_list else None,
            device=device,
            use_lora=True,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            lora_target_modules=LORA_TARGET_MODULES,
            **common_train_kwargs,
        )

    else:
        raise ValueError(f"Unknown model: {model_name!r}")


def _get_sample_kwargs(
    model_name: str,
    dataset_name: str,
    train_df: pd.DataFrame,
    n_synth: int,
    fitted_order,
) -> dict:
    cfg = DATASET_CONFIGS[dataset_name]
    base = {"n_samples": n_synth, "max_length": cfg["max_length"]}

    if model_name in ("GReaT", "PAFT"):
        # GReaT/PAFT sample() does not accept top_p
        return {**base, "k": cfg["sample_k"]}

    if model_name in ("GraDe", "GraFT"):
        # Resolve start_col: config value takes priority, then fitted_order[0],
        # then first column of train_df.
        start_col = cfg.get("start_col")
        if model_name == "GraFT" and fitted_order:
            start_col = fitted_order[0]
        if start_col is None:
            cols = fitted_order or cfg.get("train_order") or train_df.columns.tolist()
            start_col = cols[0] if cols else train_df.columns[0]

        # Always pass start_col_dist directly from train_df so the model
        # does not have to look up its internally stored distribution.
        start_col_dist = None
        if start_col in train_df.columns:
            start_col_dist = train_df[start_col].value_counts(normalize=True).to_dict()

        return {
            **base,
            "start_col": start_col,
            "start_col_dist": start_col_dist,
            "temperature": 0.7,
            "top_p": TOP_P,
            "drop_nan": False,
        }

    return base

# ── core generation function ──────────────────────────────────────────────────

def generate_one(
    model_name: str,
    dataset_name: str,
    device: str,
    paft_use_hyfd: bool,
    force: bool,
    dry_run: bool,
    ckpt_base: Path,
) -> bool:
    """Train one model on one dataset and write the synthetic CSV.

    The time-index column ``t`` is always dropped before training.
    Checkpoint behaviour mirrors generate_synthetic.py: training is skipped
    if a saved checkpoint exists, unless ``--force`` is passed.

    Returns True on success, False on failure.
    """
    out_dir = SYNTH_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_name}_synth.csv"

    if out_path.exists() and not force:
        logger.info("SKIP  %s / %s — %s already exists.", model_name, dataset_name, out_path.name)
        return True

    train_csv = ORIGIN_ROOT / dataset_name / "train.csv"
    if not train_csv.exists():
        logger.error("Training data not found: %s", train_csv)
        return False

    if dry_run:
        logger.info("DRY-RUN  %s / %s  →  %s", model_name, dataset_name, out_path)
        return True

    logger.info("=" * 64)
    logger.info("MODEL: %-8s  DATASET: %s", model_name, dataset_name)
    logger.info("=" * 64)

    train_df = pd.read_csv(train_csv)
    cfg = DATASET_CONFIGS[dataset_name]

    # Drop time-index column (always) and any extra dataset-specific columns.
    drop = ["t"] + [c for c in cfg.get("drop_cols", []) if c != "t"]
    train_df = train_df.drop(columns=[c for c in drop if c in train_df.columns])

    # Reorder for GraDe when a pre-computed IB order is available.
    if model_name == "GraDe" and cfg.get("train_order") is not None:
        order = [c for c in cfg["train_order"] if c in train_df.columns]
        train_df = train_df[order]

    n_synth = len(train_df)
    logger.info("Loaded train data: shape=%s  n_synth=%d", train_df.shape, n_synth)

    exp = dict(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        model=model_name,
        dataset=dataset_name,
        llm=LLM,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        learning_rate=LEARNING_RATE,
        train_rows=len(train_df),
        train_cols=len(train_df.columns),
        n_synth=n_synth,
        train_time_s=None,
        sample_time_s=None,
        total_time_s=None,
        synth_rows=None,
        synth_cols=None,
        gpu_train_peak_gb=None,
        gpu_train_alloc_gb=None,
        gpu_sample_peak_gb=None,
        gpu_sample_alloc_gb=None,
        gpu_total_gb=None,
        train_from_ckpt=False,
        status="FAIL",
        error="",
    )

    save_dir = _model_save_dir(ckpt_base, model_name, dataset_name)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")

            # ── Training (or load from checkpoint) ────────────────────────────
            if not force and _meta_path(save_dir).exists():
                logger.info("  Loading saved checkpoint from %s ...", save_dir)
                model, fitted_order, saved_train_time = _load_saved_model(
                    model_name, save_dir, device
                )
                exp["train_time_s"]    = saved_train_time
                exp["train_from_ckpt"] = True
                logger.info("  Checkpoint loaded (original train_time=%.1fs)", saved_train_time or 0)
            else:
                model = build_model(
                    model_name, train_df, dataset_name, device, paft_use_hyfd, ckpt_base
                )
                logger.info("  Training %s ...", model_name)
                _reset_peak()
                t_train_start = time.time()

                if model_name == "GraDe":
                    start_col = cfg.get("start_col") or train_df.columns[0]
                    model.fit(train_df, conditional_col=start_col)
                else:
                    model.fit(train_df)

                exp["train_time_s"]      = round(time.time() - t_train_start, 1)
                snap = _gpu_snapshot()
                exp["gpu_train_peak_gb"]  = snap.get("gpu_peak_gb")
                exp["gpu_train_alloc_gb"] = snap.get("gpu_alloc_gb")
                exp["gpu_total_gb"]       = snap.get("gpu_total_gb")
                logger.info(
                    "  Train done: %.1fs  GPU peak=%.2fGB alloc=%.2fGB",
                    exp["train_time_s"],
                    exp["gpu_train_peak_gb"] or 0.0,
                    exp["gpu_train_alloc_gb"] or 0.0,
                )

                fitted_order = getattr(model, "fitted_order_", None)
                _save_model(model, model_name, save_dir, fitted_order, exp["train_time_s"])

            # ── Sampling ──────────────────────────────────────────────────────
            logger.info("  Sampling %d rows ...", n_synth)
            _reset_peak()
            t_sample_start = time.time()

            sample_kwargs = _get_sample_kwargs(
                model_name, dataset_name, train_df, n_synth, fitted_order
            )
            synth_df = model.sample(**sample_kwargs)

            exp["sample_time_s"]      = round(time.time() - t_sample_start, 1)
            snap = _gpu_snapshot()
            exp["gpu_sample_peak_gb"]  = snap.get("gpu_peak_gb")
            exp["gpu_sample_alloc_gb"] = snap.get("gpu_alloc_gb")
            exp["synth_rows"] = len(synth_df)
            exp["synth_cols"] = len(synth_df.columns)

            if len(synth_df) == 0:
                raise RuntimeError(
                    "Sampling returned 0 rows. Check model training or increase epochs/max_length."
                )

            logger.info(
                "  Sample done: %.1fs  synth=%s  GPU peak=%.2fGB",
                exp["sample_time_s"], synth_df.shape,
                exp["gpu_sample_peak_gb"] or 0.0,
            )
            exp["total_time_s"] = round(
                (exp["train_time_s"] or 0) + (exp["sample_time_s"] or 0), 1
            )
            exp["status"] = "OK"

    except Exception:
        tb = traceback.format_exc()
        exp["error"] = tb.strip().splitlines()[-1][:200]
        logger.error("FAILED  %s / %s:\n%s", model_name, dataset_name, tb)
        _append_exp_log(exp)
        _flush_exp_csv()
        _free_gpu()
        return False

    synth_df.to_csv(out_path, index=False)
    logger.info("Saved  →  %s  shape=%s", out_path, synth_df.shape)

    _append_exp_log(exp)
    _flush_exp_csv()
    _free_gpu()
    return True


def run_all(
    models: list[str],
    datasets: list[str],
    device: str,
    paft_use_hyfd: bool,
    force: bool,
    dry_run: bool,
    ckpt_base: Path,
) -> dict[tuple[str, str], bool]:
    results: dict[tuple[str, str], bool] = {}
    for dataset_name in datasets:
        for model_name in models:
            ok = generate_one(
                model_name=model_name,
                dataset_name=dataset_name,
                device=device,
                paft_use_hyfd=paft_use_hyfd,
                force=force,
                dry_run=dry_run,
                ckpt_base=ckpt_base,
            )
            results[(model_name, dataset_name)] = ok

    successes = sum(v for v in results.values())
    failures  = [(m, d) for (m, d), v in results.items() if not v]
    total     = len(results)

    # ── summary table ─────────────────────────────────────────────────────────
    records = [r for r in _EXP_RECORDS]
    logger.info("")
    logger.info("━" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("━" * 80)
    if records:
        fmt = "%-7s %-12s %-6s %9s %10s %10s %10s %10s"
        logger.info(
            fmt, "Model", "Dataset", "Status",
            "Train(s)", "Sample(s)", "Total(s)", "TrainPk(GB)", "SampPk(GB)",
        )
        logger.info("-" * 80)
        for r in records:
            logger.info(
                fmt,
                r["model"], r["dataset"], r["status"],
                str(r.get("train_time_s") or "-"),
                str(r.get("sample_time_s") or "-"),
                str(r.get("total_time_s") or "-"),
                str(r.get("gpu_train_peak_gb") or "-"),
                str(r.get("gpu_sample_peak_gb") or "-"),
            )
    logger.info("━" * 80)
    logger.info("Done.  %d/%d succeeded.", successes, total)
    if failures:
        logger.warning("Failed runs:")
        for m, d in failures:
            logger.warning("  ✗  %s / %s", m, d)
    logger.info("Experiment log  →  %s", EXP_LOG_PATH)
    logger.info("Experiment CSV  →  %s", EXP_CSV_PATH)
    logger.info("━" * 80)
    logger.info("Next step:  python run_benchmark2.py")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 2: train models on concept-drift datasets and generate synthetic data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
        metavar="MODEL", help="Models to run.  Choices: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=ALL_DATASETS, default=ALL_DATASETS,
        metavar="DATASET", help="Datasets to run.  Choices: " + ", ".join(ALL_DATASETS),
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Compute device.",
    )
    parser.add_argument(
        "--paft-use-hyfd", action=argparse.BooleanOptionalAction, default=True,
        help="Auto-discover PAFT column order via HyFD (default: on).",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Overwrite existing synthetic CSVs and re-train from scratch.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print what would run without actually training.",
    )
    parser.add_argument(
        "--ckpt-base", type=Path, default=CKPT_BASE,
        help="Root directory for model checkpoints.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(
        models=args.models,
        datasets=args.datasets,
        device=args.device,
        paft_use_hyfd=args.paft_use_hyfd,
        force=args.force,
        dry_run=args.dry_run,
        ckpt_base=args.ckpt_base,
    )

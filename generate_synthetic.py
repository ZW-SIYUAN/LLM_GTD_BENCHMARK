"""
generate_synthetic.py — Train models and generate synthetic CSV files.
======================================================================
Trains each of the four generative models (GReaT, PAFT, GraDe, GraFT) on
every dataset and writes the synthetic samples to:

    llm_gtd_benchmark/data/synthetic/<ModelName>/<dataset>_synth.csv

After generation, run ``python run_benchmark.py`` to evaluate all files.

Usage
-----
# All models × all datasets (default)
python generate_synthetic.py

# Specific model / dataset
python generate_synthetic.py --models GReaT PAFT --datasets income diabetes

# Dry-run: print what would run without training
python generate_synthetic.py --dry-run

# Skip completed CSV files (default behaviour; --force overwrites)
python generate_synthetic.py --force

GPU / CPU selection
-------------------
By default the script uses CUDA if available.  Pass ``--device cpu`` to
force CPU (useful for debugging or small datasets).

HyFD FD order (PAFT only)
--------------------------
When ``--paft-use-hyfd`` is set (default), PAFT's column order is
auto-discovered from the training data via HyFD before training.
Disable with ``--no-paft-use-hyfd`` to fall back to random permutation.
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
# Suppress PyTorch pin_memory DeprecationWarning from DataLoader worker subprocesses.
# filterwarnings() alone is insufficient because worker processes don't inherit it;
# PYTHONWARNINGS env var IS inherited by subprocesses.
os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore::DeprecationWarning:torch.utils.data._utils.pin_memory",
)
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*is_pinned.*", category=DeprecationWarning)
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
logger = logging.getLogger("generate_synthetic")

# ── experiment logging helpers ────────────────────────────────────────────────

EXP_LOG_PATH = ROOT / "experiment_log.jsonl"
EXP_CSV_PATH = ROOT / "experiment_log.csv"


def _gpu_snapshot() -> dict:
    """Return current / peak GPU memory stats (GB). Empty dict if no GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        torch.cuda.synchronize()
        return {
            "gpu_alloc_gb":  round(torch.cuda.memory_allocated()  / 1024**3, 3),
            "gpu_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
            "gpu_peak_gb":   round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "gpu_total_gb":  round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        }
    except Exception:
        return {}


def _reset_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _free_gpu():
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _append_exp_log(record: dict) -> None:
    """Append one JSON record to experiment_log.jsonl (one line per run)."""
    with open(EXP_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _flush_exp_csv() -> None:
    """Re-write experiment_log.csv from all lines in experiment_log.jsonl."""
    if not EXP_LOG_PATH.exists():
        return
    records = []
    with open(EXP_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if records:
        pd.DataFrame(records).to_csv(EXP_CSV_PATH, index=False)


# ── paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = ROOT / "llm_gtd_benchmark" / "data"
ORIGIN_ROOT = DATA_ROOT / "origin_data"
SYNTH_ROOT  = DATA_ROOT / "synthetic"

# ── global hyper-parameters ───────────────────────────────────────────────────
LLM           = "gpt2-medium"
LEARNING_RATE = 5e-5
TEMPERATURE   = 0.8
TOP_P         = 1.0
SEED          = 42

# LoRA
LORA_R              = 8
LORA_ALPHA          = 16
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["c_attn", "c_proj"]


# ── per-dataset configs ────────────────────────────────────────────────────────
#
#  epochs        : fine-tuning epochs
#  batch_size    : per-device training batch size
#  max_length    : max token length per generated row
#  sample_k      : top-k for legacy sampling (controls KV-cache peak memory)
#  start_col     : GraDe/GraFT generation starting column
#  train_order   : GraDe/GraFT fixed column order (IB-optimal, from example runs)
#  fd_list       : GraDe/GraFT FD constraints (index into train_order, 0-based)
#  drop_cols     : columns to drop before training (e.g. meaningless IDs)
#
DATASET_CONFIGS: dict[str, dict] = {

    "diabetes": {
        # IB order: Glucose → DiabetesPedigreeFunction → BMI → Age → …
        "train_order": [
            "Glucose", "DiabetesPedigreeFunction", "BMI", "Age",
            "BloodPressure", "Insulin", "SkinThickness", "Pregnancies", "Diabetes",
        ],
        # 0:Glucose 1:DPF 2:BMI 3:Age 4:BP 5:Insulin 6:SkinThickness 7:Pregnancies 8:Diabetes
        "fd_list": [
            [[0], [8]],   # Glucose      → Diabetes
            [[2], [8]],   # BMI          → Diabetes
            [[3], [7]],   # Age          → Pregnancies
        ],
        "start_col":  "Glucose",
        "epochs":     100,
        "batch_size": 32,
        "max_length": 100,
        "sample_k":   100,
        "drop_cols":  [],
    },

    "house": {
        # IB order: total_rooms → population → median_income → …
        "train_order": [
            "total_rooms", "population", "median_income", "total_bedrooms",
            "households", "longitude", "latitude",
            "median_house_value", "housing_median_age", "ocean_proximity",
        ],
        # 0:total_rooms 1:population 2:median_income 3:total_bedrooms
        # 4:households 5:longitude 6:latitude 7:median_house_value
        # 8:housing_median_age 9:ocean_proximity
        "fd_list": [
            [[0], [3]],   # total_rooms   → total_bedrooms
            [[0], [4]],   # total_rooms   → households
            [[2], [7]],   # median_income → median_house_value
        ],
        "start_col":  "total_rooms",
        "epochs":     100,
        "batch_size": 16,
        "max_length": 150,
        "sample_k":   50,
        "drop_cols":  [],
    },

    "income": {
        # IB order: fnlwgt → age → hours-per-week → occupation → …
        "train_order": [
            "fnlwgt", "age", "hours-per-week", "occupation",
            "education", "education-num", "relationship", "marital-status",
            "workclass", "gender", "capital-gain", "native-country",
            "income", "race", "capital-loss",
        ],
        # 0:fnlwgt 1:age 2:hours-per-week 3:occupation 4:education 5:education-num
        # 6:relationship 7:marital-status 8:workclass 9:gender
        # 10:capital-gain 11:native-country 12:income 13:race 14:capital-loss
        "fd_list": [
            [[4],  [5]],   # education    → education-num
            [[6],  [7]],   # relationship → marital-status
            [[4],  [12]],  # education    → income
            [[3],  [12]],  # occupation   → income
        ],
        "start_col":  "fnlwgt",
        "epochs":     60,
        "batch_size": 16,
        "max_length": 200,
        "sample_k":   20,
        "drop_cols":  [],
    },

    "sick": {
        # IB order: TT4 → FTI → age → TSH → T4U → T3 → referral_source → sex → …
        "train_order": [
            "TT4", "FTI", "age", "TSH", "T4U", "T3",
            "referral_source", "sex",
            "T3_measured", "on_thyroxine", "T4U_measured", "FTI_measured",
            "TSH_measured", "Class",
            "query_hyperthyroid", "query_hypothyroid", "TT4_measured",
            "psych", "sick", "tumor", "I131_treatment",
            "query_on_thyroxine", "pregnant", "thyroid_surgery",
            "on_antithyroid_medication", "goitre", "lithium",
            "hypopituitary", "TBG", "TBG_measured",
        ],
        # FDs: lab value → its measured flag
        # 0:TT4 1:FTI 3:TSH 4:T4U 5:T3
        # 8:T3_measured 10:T4U_measured 11:FTI_measured 12:TSH_measured 16:TT4_measured
        "fd_list": [
            [[5],  [8]],   # T3  → T3_measured
            [[4],  [10]],  # T4U → T4U_measured
            [[1],  [11]],  # FTI → FTI_measured
            [[3],  [12]],  # TSH → TSH_measured
            [[0],  [16]],  # TT4 → TT4_measured
        ],
        "start_col":  "TT4",
        "epochs":     100,
        "batch_size": 4,
        "max_length": 400,
        "sample_k":   10,
        "drop_cols":  [],
    },

    "us_location": {
        # No pre-computed IB order available; GraDe/GraFT use auto-discovery at runtime.
        "train_order": None,
        "fd_list":     [],
        "start_col":   None,  # resolved to the first column of train_order at runtime
        "epochs":     100,
        "batch_size": 32,
        "max_length": 100,
        "sample_k":   100,
        "drop_cols":  [],
    },
}

ALL_DATASETS = list(DATASET_CONFIGS.keys())
ALL_MODELS   = ["GReaT", "PAFT", "GraDe", "GraFT"]


# ── model factory ─────────────────────────────────────────────────────────────

def build_model(
    model_name: str,
    train_df: pd.DataFrame,
    dataset_name: str,
    device: str,
    paft_use_hyfd: bool,
    ckpt_base: Path,
):
    """Instantiate and return the requested model wrapper.

    Parameters
    ----------
    model_name:
        One of ``"GReaT"``, ``"PAFT"``, ``"GraDe"``, ``"GraFT"``.
    train_df:
        Training DataFrame (used by PAFT for HyFD discovery and by
        GraDe/GraFT to resolve ``start_col`` when not pre-configured).
    dataset_name:
        Used to namespace the Trainer checkpoint directory.
    device:
        ``"cuda"`` or ``"cpu"``.
    paft_use_hyfd:
        When True, PAFT's column order is discovered via HyFD before training.
    ckpt_base:
        Root directory for Trainer checkpoints.
    """
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
    n_samples: int,
    fitted_order=None,
) -> dict:
    """Build keyword arguments for the model's ``sample()`` call."""
    cfg        = DATASET_CONFIGS[dataset_name]
    k          = cfg.get("sample_k")
    max_length = cfg.get("max_length")

    if model_name in ("GReaT", "PAFT"):
        kwargs: dict = dict(n_samples=n_samples, temperature=TEMPERATURE)
        if k is not None:
            kwargs["k"] = k
        if max_length is not None:
            kwargs["max_length"] = max_length
        return kwargs

    # GraDe / GraFT — resolve start_col
    start_col = cfg.get("start_col")

    # GraFT auto-discovers order; use the fitted first column as start
    if model_name == "GraFT" and fitted_order:
        start_col = fitted_order[0]

    # Fallback: first column of the resolved order
    if start_col is None:
        cols = (
            fitted_order
            or cfg.get("train_order")
            or train_df.columns.tolist()
        )
        start_col = cols[0] if cols else train_df.columns[0]

    start_col_dist = None
    if start_col in train_df.columns:
        start_col_dist = train_df[start_col].value_counts(normalize=True).to_dict()

    kwargs = dict(
        n_samples=n_samples,
        start_col=start_col,
        start_col_dist=start_col_dist,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        drop_nan=False,
    )
    if k is not None:
        kwargs["k"] = k
    if max_length is not None:
        kwargs["max_length"] = max_length
    return kwargs


# ── generation loop ───────────────────────────────────────────────────────────

def _model_save_dir(ckpt_base: Path, model_name: str, dataset_name: str) -> Path:
    """Return the directory where a completed model is saved."""
    return ckpt_base / model_name / f"{dataset_name}_saved"


def _meta_path(save_dir: Path) -> Path:
    return save_dir / ".meta.json"


def _merge_lora(model) -> None:
    """Merge LoRA adapter weights into the base model in-place.

    GReaT / PAFT / GraDe / GraFT all store the inner backbone as
    ``model._model.model`` (a PeftModel after _apply_lora()).
    Both ``save()`` implementations call ``torch.save(self.model.state_dict(),
    "model.pt")`` and ``load_from_dir()`` does ``base.load_state_dict(...)``
    on a freshly constructed base architecture — so the saved state_dict must
    use the base model's key layout, not PeftModel's adapter-only keys.
    Merging folds the LoRA delta back into the base weights and returns a
    plain transformer that is compatible with ``load_state_dict``.
    """
    inner = getattr(model, "_model", None)          # GReaTModel._model  → GReaT instance
    if inner is None:
        return
    lm = getattr(inner, "model", None)              # GReaT.model        → PeftModel (after LoRA)
    if lm is None:
        return
    if hasattr(lm, "merge_and_unload"):
        inner.model = lm.merge_and_unload()
        logger.info("  LoRA weights merged into base model for saving.")


def _save_model(model, model_name: str, save_dir: Path, fitted_order, train_time_s) -> None:
    """Merge LoRA, save model weights + metadata sidecar after successful training."""
    _merge_lora(model)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(save_dir))
    meta = {
        "model": model_name,
        "fitted_order": fitted_order,
        "train_time_s": train_time_s,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(_meta_path(save_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("  Checkpoint saved  →  %s", save_dir)


def _load_saved_model(model_name: str, save_dir: Path, device: str):
    """Load a completed model from save_dir. Returns (model, fitted_order, train_time_s)."""
    meta_file = _meta_path(save_dir)
    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)
    fitted_order = meta.get("fitted_order")
    train_time_s = meta.get("train_time_s")

    if model_name == "GReaT":
        model = GReaTModel.load(str(save_dir))
    elif model_name == "PAFT":
        model = PAFTModel.load(str(save_dir))
        model._fitted_order = fitted_order
    elif model_name == "GraDe":
        model = GraDeModel.load(str(save_dir), device=device)
    elif model_name == "GraFT":
        model = GraFTModel.load(str(save_dir), device=device)
        model.fitted_order_ = fitted_order
    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    return model, fitted_order, train_time_s


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

    Checkpoint behaviour
    --------------------
    After training succeeds, model weights are saved to
    ``<ckpt_base>/<model>/<dataset>_saved/`` alongside a ``.meta.json``
    sidecar that stores ``fitted_order`` and ``train_time_s``.

    On the next run (e.g. after a sampling crash), training is **skipped**
    and the saved weights are loaded directly — unless ``--force`` is set.

    Returns True on success, False on failure/skip.
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

    # Drop configured columns
    for c in cfg.get("drop_cols", []):
        if c in train_df.columns:
            train_df = train_df.drop(columns=[c])

    # Reorder columns for GraDe (GraFT handles reordering internally via IB)
    if model_name == "GraDe" and cfg.get("train_order") is not None:
        order = [c for c in cfg["train_order"] if c in train_df.columns]
        train_df = train_df[order]

    n_synth = len(train_df)
    logger.info("Loaded train data: shape=%s  n_synth=%d", train_df.shape, n_synth)

    # ── experiment record (filled progressively) ──────────────────────────────
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
                exp["train_time_s"]      = saved_train_time
                exp["train_from_ckpt"]   = True
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

                exp["train_time_s"] = round(time.time() - t_train_start, 1)
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

            exp["sample_time_s"] = round(time.time() - t_sample_start, 1)
            snap = _gpu_snapshot()
            exp["gpu_sample_peak_gb"]  = snap.get("gpu_peak_gb")
            exp["gpu_sample_alloc_gb"] = snap.get("gpu_alloc_gb")
            exp["synth_rows"] = len(synth_df)
            exp["synth_cols"] = len(synth_df.columns)
            logger.info(
                "  Sample done: %.1fs  synth=%s  GPU peak=%.2fGB",
                exp["sample_time_s"],
                synth_df.shape,
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

    # ── print timing / memory summary from log ────────────────────────────────
    logger.info("")
    logger.info("━" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("━" * 80)
    if EXP_LOG_PATH.exists():
        records = []
        with open(EXP_LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        # Keep only the latest record per (model, dataset) for this run
        run_keys = {(m, d) for (m, d) in results}
        seen: dict = {}
        for r in records:
            key = (r["model"], r["dataset"])
            if key in run_keys:
                seen[key] = r          # later records overwrite earlier ones
        records = list(seen.values())
        if records:
            fmt = "%-7s %-12s %-6s %9s %10s %10s %10s %10s"
            logger.info(
                fmt,
                "Model", "Dataset", "Status",
                "Train(s)", "Sample(s)", "Total(s)",
                "TrainPk(GB)", "SampPk(GB)",
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
    logger.info(
        "Next step:  python run_benchmark.py  (evaluates all generated CSVs)"
    )
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train models and generate synthetic tabular datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=ALL_MODELS,
        metavar="MODEL",
        help="Models to run.  Choices: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        default=ALL_DATASETS,
        metavar="DATASET",
        help="Datasets to run.  Choices: " + ", ".join(ALL_DATASETS),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for GraDe / GraFT sampling.",
    )
    parser.add_argument(
        "--paft-use-hyfd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-discover PAFT column order via HyFD (default: on).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing synthetic CSVs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would run without actually training.",
    )
    parser.add_argument(
        "--ckpt-dir",
        default=str(ROOT / "checkpoints"),
        help="Root directory for Trainer checkpoints.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available — switching to CPU.")
                args.device = "cpu"
        except ImportError:
            logger.warning("torch not importable — defaulting to CPU.")
            args.device = "cpu"

    run_all(
        models=args.models,
        datasets=args.datasets,
        device=args.device,
        paft_use_hyfd=args.paft_use_hyfd,
        force=args.force,
        dry_run=args.dry_run,
        ckpt_base=Path(args.ckpt_dir),
    )

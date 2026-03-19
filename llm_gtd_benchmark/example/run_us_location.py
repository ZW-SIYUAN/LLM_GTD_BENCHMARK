"""
GraDe Experiment: us_location dataset
  llm          : gpt2-medium
  lora         : r=8, alpha=16, dropout=0.05
  precision    : bf16
  epochs       : 100
  batch_size   : 32
  temperature  : 0.8
  max_length   : 50
  sample_k     : 100
  sampling     : legacy_sample (standard autoregressive, model.sample())

Columns:
    0: state_code  (categorical)
    1: lat         (continuous)
    2: lon         (continuous)
    3: bird        (categorical)
    4: lat_zone    (categorical)

Functional Dependencies:
    state_code → bird       [[0], [3]]
    state_code → lat_zone   [[0], [4]]
    lat        → lat_zone   [[1], [4]]
    bird       → state_code [[3], [0]]
"""

import os
import sys
import logging
import time
import random

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grade import GraDe

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("us_location_experiment.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = os.path.join("data", "us_location")
TRAIN_PATH     = os.path.join(DATA_DIR, "train.csv")
CHECKPOINT_DIR = "checkpoints_us_location_fixed"   # 新目录，不覆盖上次随机顺序的结果
OUTPUT_PATH    = "synthetic_us_location_fixed.csv"

# ── hyperparameters ───────────────────────────────────────────────────────────
LLM             = "gpt2-medium"
EPOCHS          = 100
BATCH_SIZE      = 32
LEARNING_RATE   = 5e-5
SPARSITY_LAMBDA = 0.001
FD_LAMBDA       = 0.1
FD_ALPHA        = 0.5
NUM_HEAD_GROUPS = 4
# sampling
TEMPERATURE     = 0.8
MAX_LENGTH      = 50
SAMPLE_K        = 100          # batch size per generate() call
TOP_P           = 1.0          # legacy_sample: no nucleus filtering
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# LoRA
LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05
# GPT-2 attention uses a single fused c_attn (Q+K+V) and output projection c_proj
LORA_TARGET_MODULES = ["c_attn", "c_proj"]

# bf16: supported on Ampere+ (A100, RTX 3090, etc.)
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16 = False
if not USE_BF16:
    logger.warning("bf16 not supported on this device, falling back to fp32.")

# ── fixed column order ────────────────────────────────────────────────────────
TRAIN_ORDER = ["lon", "lat", "state_code", "bird", "lat_zone"]

# ── functional dependencies ───────────────────────────────────────────────────
FD_LIST = [
    [[2], [3]],   # state_code(2) → bird(3)
    [[2], [4]],   # state_code(2) → lat_zone(4)
    [[1], [4]],   # lat(1)        → lat_zone(4)
    [[3], [2]],   # bird(3)       → state_code(2)
]


# ── LoRA helper ───────────────────────────────────────────────────────────────
def apply_lora(model):
    """Wrap model.model (TabDynamicGraphGPT2) with LoRA via PEFT."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        logger.error("PEFT not installed. Run: pip install peft")
        raise

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        # exclude graph_generator: it's a new module trained from scratch,
        # LoRA is for adapting pre-trained weights only
        modules_to_save=None,
        bias="none",
    )

    model.model = get_peft_model(model.model, lora_config)
    model.model.print_trainable_parameters()
    logger.info("LoRA applied: r=%d  alpha=%d  dropout=%.2f  targets=%s",
                LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES)
    return model


# ── data ──────────────────────────────────────────────────────────────────────
def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    logger.info("Train shape: %s  |  columns: %s",
                train_df.shape, train_df.columns.tolist())
    logger.info("Sample:\n%s", train_df.head(3).to_string())
    for col in train_df.columns:
        if train_df[col].dtype == "object" or train_df[col].nunique() < 30:
            logger.info("  [%s] %d unique  top5=%s",
                        col, train_df[col].nunique(),
                        train_df[col].value_counts().head(5).to_dict())
        else:
            logger.info("  [%s] min=%.3f  max=%.3f  mean=%.3f",
                        col, train_df[col].min(), train_df[col].max(), train_df[col].mean())
    return train_df


# ── model ─────────────────────────────────────────────────────────────────────
def build_model():
    logger.info("Building GraDe  |  llm=%s  bf16=%s  LoRA=r%d",
                LLM, USE_BF16, LORA_R)
    model = GraDe(
        llm=LLM,
        experiment_dir=CHECKPOINT_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        sparsity_lambda=SPARSITY_LAMBDA,
        use_dynamic_graph=True,
        num_head_groups=NUM_HEAD_GROUPS,
        fd_lambda=FD_LAMBDA,
        fd_alpha=FD_ALPHA,
        fd_list=FD_LIST,
        fixed_col_order=True,          # 固定列顺序训练+采样
        save_steps=500,
        logging_steps=100,
        bf16=USE_BF16,
        fp16=USE_FP16,
    )
    model = apply_lora(model)
    return model


# ── train ─────────────────────────────────────────────────────────────────────
def train(model, train_df):
    logger.info("Training  |  %d samples  |  %d epochs  |  device=%s",
                len(train_df), EPOCHS, DEVICE)
    logger.info("Column order: %s", TRAIN_ORDER)
    train_df = train_df[TRAIN_ORDER]   # 按指定顺序重排列
    t0 = time.time()
    model.fit(train_df, conditional_col="state_code", fd_list=FD_LIST)
    elapsed = time.time() - t0
    logger.info("Training done  |  %.1f s (%.1f min)", elapsed, elapsed / 60)


# ── sample ────────────────────────────────────────────────────────────────────
def sample(model, n_samples):
    logger.info("Sampling %d rows  |  legacy_sample  |  T=%.2f  k=%d  max_len=%d",
                n_samples, TEMPERATURE, SAMPLE_K, MAX_LENGTH)
    t0 = time.time()
    synthetic_df = model.sample(
        n_samples=n_samples,
        start_col="state_code",
        start_col_dist=model.conditional_col_dist,
        temperature=TEMPERATURE,
        k=SAMPLE_K,
        top_p=TOP_P,
        max_length=MAX_LENGTH,
        drop_nan=False,
        device=DEVICE,
    )
    elapsed = time.time() - t0
    logger.info("Sampling done  |  %.1f s  |  got %d / %d rows",
                elapsed, len(synthetic_df), n_samples)
    return synthetic_df


# ── fidelity check ────────────────────────────────────────────────────────────
def compare_stats(real_df, syn_df):
    logger.info("── fidelity check ──────────────────────────────")
    for col in real_df.columns:
        if col not in syn_df.columns:
            logger.warning("  [%s] missing in synthetic data", col)
            continue
        if real_df[col].dtype == "object" or real_df[col].nunique() < 30:
            r_top = set(real_df[col].value_counts().head(5).index)
            s_top = set(syn_df[col].value_counts().head(5).index)
            logger.info("  [%s] top-5 overlap %d/5  real=%s  syn=%s",
                        col, len(r_top & s_top), list(r_top), list(s_top))
        else:
            s_num = pd.to_numeric(syn_df[col], errors="coerce")
            logger.info("  [%s] real mean=%.3f std=%.3f  |  syn mean=%.3f std=%.3f",
                        col,
                        real_df[col].mean(), real_df[col].std(),
                        s_num.mean(), s_num.std())


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("GraDe  ·  us_location  ·  gpt2-medium + LoRA")
    logger.info("=" * 60)

    train_df = load_data()
    n_samples = len(train_df)   # 16320

    model = build_model()
    train(model, train_df)

    synthetic_df = sample(model, n_samples)
    synthetic_df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved: %s", OUTPUT_PATH)

    compare_stats(train_df, synthetic_df)
    logger.info("=" * 60)
    logger.info("Done.  synthetic rows: %d / %d", len(synthetic_df), n_samples)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

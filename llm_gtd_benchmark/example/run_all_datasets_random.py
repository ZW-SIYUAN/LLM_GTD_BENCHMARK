"""
GraDe Experiment Runner — 6 datasets (excluding us_location)
  llm        : gpt2-medium
  lora       : r=8, alpha=16, dropout=0.05
  precision  : bf16 (falls back to fp32)
  col_order  : random (shuffle per sample, per epoch)
  sampling   : legacy_sample (model.sample)

Usage:
    python run_all_datasets.py                # run all 6 datasets
    python run_all_datasets.py bird           # run a single dataset
    python run_all_datasets.py bird diabetes  # run multiple
"""

import os
import sys
import logging
import time
import random
import argparse

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grade import GraDe

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── global hyper-params (shared across datasets) ───────────────────────────────
LLM             = "gpt2-medium"
BATCH_SIZE      = 32
LEARNING_RATE   = 5e-5
SPARSITY_LAMBDA = 0.001
FD_LAMBDA       = 0.1
FD_ALPHA        = 0.5
NUM_HEAD_GROUPS = 4
TEMPERATURE     = 0.8
SAMPLE_K        = 100
TOP_P           = 1.0
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16        = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

LORA_R               = 8
LORA_ALPHA_LORA      = 16
LORA_DROPOUT         = 0.05
LORA_TARGET_MODULES  = ["c_attn", "c_proj"]


# ── per-dataset configs ────────────────────────────────────────────────────────
#
#  train_order  : fixed column order during training AND sampling
#  fd_list      : [[left_col_indices], [right_col_indices]]  (0-based in train_order)
#  start_col    : first column in train_order (used as sampling start)
#  epochs       : adjusted by dataset size to target ~50 000 steps
#  max_length   : rough estimate: ~8 tokens/col × n_cols + buffer
#  drop_cols    : columns to drop before everything (e.g. meaningless IDs)
#
DATASETS = {

    "bird": {
        # IB order: lon → lat → state_code → bird → lat_zone
        # chain_CE=8.2743  chain_MI=8.8545
        # peak=5.21GB ✅
        "train_order": ["lon", "lat", "state_code", "bird", "lat_zone"],
        # 0:lon  1:lat  2:state_code  3:bird  4:lat_zone
        "fd_list": [
            [[2], [3]],   # state_code → bird
            [[2], [4]],   # state_code → lat_zone
            [[1], [4]],   # lat        → lat_zone
            [[3], [2]],   # bird       → state_code  (mutual)
        ],
        "start_col":  "lon",
        "epochs":     100,     # 12869/32 ≈ 402 steps/epoch → 40 200 steps
        "max_length": 50,
        "batch_size": 32,
        "sample_k":   100,
        "drop_cols":  [],
    },

    "diabetes": {
        # IB order: Glucose → DiabetesPedigreeFunction → BMI → Age →
        #           BloodPressure → Insulin → SkinThickness → Pregnancies → Diabetes
        # chain_CE=19.4682  chain_MI=17.0740
        # peak=6.96GB ✅
        "train_order": [
            "Glucose", "DiabetesPedigreeFunction", "BMI", "Age",
            "BloodPressure", "Insulin", "SkinThickness", "Pregnancies", "Diabetes",
        ],
        # 0:Glucose 1:DPF 2:BMI 3:Age 4:BP 5:Insulin 6:SkinThickness 7:Pregnancies 8:Diabetes
        "fd_list": [
            [[0], [8]],   # Glucose → Diabetes
            [[2], [8]],   # BMI     → Diabetes
            [[3], [7]],   # Age     → Pregnancies
        ],
        "start_col":  "Glucose",
        "epochs":     100,     # 614/32 ≈ 19 steps/epoch → 1 900 steps (small dataset)
        "max_length": 100,
        "batch_size": 32,
        "sample_k":   100,
        "drop_cols":  [],
    },

    "house": {
        # IB order: total_rooms → population → median_income → total_bedrooms →
        #           households → longitude → latitude → median_house_value →
        #           housing_median_age → ocean_proximity
        # chain_CE=43.5748  chain_MI=9.9421
        # peak=10.16GB ⚠️  sample_k reduced to 50 to limit KV-cache during sampling
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
        "epochs":     100,     # 16512/16 ≈ 1032 steps/epoch → 103 200 steps
        "max_length": 150,
        "batch_size": 16,      # reduced: 10col×150tok sequences are long
        "sample_k":   30,
        "drop_cols":  [],
    },

    "income": {
        # IB order: fnlwgt → age → hours-per-week → occupation → education →
        #           education-num → relationship → marital-status → workclass →
        #           gender → capital-gain → native-country → income → race → capital-loss
        # chain_CE=23.3040  chain_MI=5.2284
        # peak=11.52GB(train) + 采样OOM → batch_size=16, sample_k=20
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
            [[4], [5]],    # education     → education-num  (near-perfect FD)
            [[6], [7]],    # relationship  → marital-status
            [[4], [12]],   # education     → income
            [[3], [12]],   # occupation    → income
        ],
        "start_col":  "fnlwgt",
        "epochs":     60,      # 26049/16 ≈ 1628 steps/epoch → 97 680 steps
        "max_length": 200,
        "batch_size": 16,      # halved: 11.52→~6GB peak
        "sample_k":   20,      # 15col×200tok KV-cache 很大，减小并发生成量
        "drop_cols":  [],
    },

    "loan": {
        # IB order: CITY → Income → Age → Profession → Experience → STATE →
        #           CURRENT_JOB_YRS → CURRENT_HOUSE_YRS → Car_Ownership →
        #           Risk_Flag → Married/Single → House_Ownership
        # chain_CE=32.3999  chain_MI=2.8573
        # peak=10.05GB ⚠️  sample_k reduced to 50
        "train_order": [
            "CITY", "Income", "Age", "Profession", "Experience", "STATE",
            "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS",
            "Car_Ownership", "Risk_Flag", "Married/Single", "House_Ownership",
        ],
        # 0:CITY 1:Income 2:Age 3:Profession 4:Experience 5:STATE
        # 6:CURRENT_JOB_YRS 7:CURRENT_HOUSE_YRS 8:Car_Ownership
        # 9:Risk_Flag 10:Married/Single 11:House_Ownership
        "fd_list": [
            [[0], [5]],    # CITY       → STATE
            [[0], [1]],    # CITY       → Income
            [[2], [4]],    # Age        → Experience
        ],
        "start_col":  "CITY",
        "epochs":     10,      # 201600/32 ≈ 6300 steps/epoch → 63 000 steps
        "max_length": 200,
        "batch_size": 16,
        "sample_k":   30,
        "drop_cols":  ["Id"],
    },

    "sick": {
        # IB order: TT4 → FTI → age → TSH → T4U → T3 → referral_source → sex →
        #           T3_measured → on_thyroxine → T4U_measured → FTI_measured →
        #           TSH_measured → Class → query_hyperthyroid → query_hypothyroid →
        #           TT4_measured → psych → sick → tumor → I131_treatment →
        #           query_on_thyroxine → pregnant → thyroid_surgery →
        #           on_antithyroid_medication → goitre → lithium → hypopituitary →
        #           TBG → TBG_measured
        # chain_CE=29.4739  chain_MI=7.2220
        # peak=20.20GB ❌ OOM  → batch_size=4, sample_k=10
        # Note: IB puts lab values first (high entropy), measured-flags after
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
        # FDs: lab value → its measured flag  (value comes before flag in IB order)
        # 0:TT4 1:FTI 2:age 3:TSH 4:T4U 5:T3
        # 8:T3_measured 10:T4U_measured 11:FTI_measured 12:TSH_measured 16:TT4_measured
        "fd_list": [
            [[5],  [8]],   # T3  → T3_measured
            [[4],  [10]],  # T4U → T4U_measured
            [[1],  [11]],  # FTI → FTI_measured
            [[3],  [12]],  # TSH → TSH_measured
            [[0],  [16]],  # TT4 → TT4_measured
        ],
        "start_col":  "TT4",
        "epochs":     100,     # 3017/4 ≈ 754 steps/epoch → 75 400 steps
        "max_length": 400,
        "batch_size": 4,       # 20.20GB→~2.5GB: 30col长序列只能小batch
        "sample_k":   10,      # 30col×400tok KV-cache极大，一次只生成10行
        "drop_cols":  [],
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────
def make_logger(name: str) -> logging.Logger:
    log_file = f"{name}_random_experiment.log"
    logger = logging.getLogger(f"{name}_random")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def apply_lora(model, logger):
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        logger.error("PEFT not installed: pip install peft")
        raise
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA_LORA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=None,
        bias="none",
    )
    model.model = get_peft_model(model.model, lora_config)
    model.model.print_trainable_parameters()
    return model


def compare_stats(real_df: pd.DataFrame, syn_df: pd.DataFrame, logger):
    logger.info("── fidelity check ──────────────────────────────")
    for col in real_df.columns:
        if col not in syn_df.columns:
            logger.warning("  [%s] missing in synthetic", col)
            continue
        if real_df[col].dtype == "object" or real_df[col].nunique() < 30:
            r_top = set(real_df[col].value_counts().head(5).index)
            s_top = set(syn_df[col].value_counts().head(5).index) if len(syn_df) else set()
            logger.info("  [%s] top-5 overlap %d/5  real=%s  syn=%s",
                        col, len(r_top & s_top), sorted(r_top), sorted(s_top))
        else:
            s_num = pd.to_numeric(syn_df[col], errors="coerce")
            logger.info("  [%s] real mean=%.3f std=%.3f  |  syn mean=%.3f std=%.3f",
                        col,
                        real_df[col].mean(), real_df[col].std(),
                        s_num.mean(), s_num.std())


# ── GPU memory reporter ───────────────────────────────────────────────────────
def report_gpu(tag: str):
    if not torch.cuda.is_available():
        return
    alloc  = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved()  / 1024**3
    total  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  [GPU {tag}]  alloc={alloc:.2f}GB  reserved={reserv:.2f}GB  total={total:.2f}GB")


# ── per-dataset runner ────────────────────────────────────────────────────────
def find_latest_checkpoint(checkpoint_dir: str):
    """Return path to the highest-numbered checkpoint-N subfolder, or None."""
    if not os.path.isdir(checkpoint_dir):
        return None
    subs = [
        d for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if not subs:
        return None
    subs.sort(key=lambda x: int(x.split("-")[1]))
    return os.path.join(checkpoint_dir, subs[-1])


def run_dataset(name: str, cfg: dict, test_mode: bool = False, sample_only: bool = False):
    logger = make_logger(name)
    logger.info("=" * 60)
    suffix = "  [TEST MODE]" if test_mode else ("  [SAMPLE-ONLY]" if sample_only else "")
    logger.info("GraDe · %s · %s + LoRA%s", name, LLM, suffix)
    logger.info("=" * 60)

    # ── paths
    train_path     = os.path.join("data", name, "train.csv")
    checkpoint_dir = f"checkpoints_{name}_random_test" if test_mode else f"checkpoints_{name}_random"
    output_path    = f"synthetic_{name}_random_test.csv" if test_mode else f"synthetic_{name}_random.csv"

    # ── load data
    train_df = pd.read_csv(train_path)
    for c in cfg["drop_cols"]:
        if c in train_df.columns:
            train_df = train_df.drop(columns=[c])

    train_order = cfg["train_order"]
    missing = [c for c in train_order if c not in train_df.columns]
    if missing:
        logger.error("Columns missing in %s: %s", name, missing)
        return
    train_df = train_df[train_order]

    n_samples  = 5   if test_mode else len(train_df)
    epochs     = 1   if test_mode else cfg["epochs"]
    batch_size = cfg.get("batch_size", BATCH_SIZE)
    sample_k   = cfg.get("sample_k",   SAMPLE_K)

    logger.info("Loaded %d rows  |  cols=%s", len(train_df), train_order)

    # ── build model skeleton
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    report_gpu("before model load")

    extra = {"max_steps": 3} if test_mode else {}
    model = GraDe(
        llm=LLM,
        experiment_dir=checkpoint_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=LEARNING_RATE,
        sparsity_lambda=SPARSITY_LAMBDA,
        use_dynamic_graph=True,
        num_head_groups=NUM_HEAD_GROUPS,
        fd_lambda=FD_LAMBDA,
        fd_alpha=FD_ALPHA,
        fd_list=cfg["fd_list"],
        fixed_col_order=False,
        save_steps=999999 if test_mode else 500,
        logging_steps=1   if test_mode else 100,
        bf16=USE_BF16,
        fp16=False,
        **extra,
    )

    if sample_only:
        # ── load adapter from latest checkpoint, skip training
        from peft import PeftModel
        ckpt = find_latest_checkpoint(checkpoint_dir)
        if ckpt is None:
            logger.error("No checkpoint found in %s — cannot use --sample-only", checkpoint_dir)
            return
        logger.info("Loading LoRA adapter from: %s", ckpt)
        model.model = PeftModel.from_pretrained(model.model, ckpt, is_trainable=False)
        model.model.eval()
        # restore metadata that fit() would have set
        model.columns      = train_df.columns.tolist()
        model.num_cols     = train_df.select_dtypes(include=np.number).columns.tolist()
        model.column_names = train_df.columns.tolist()
        logger.info("Adapter loaded, skipping training")
    else:
        model = apply_lora(model, logger)
        report_gpu("after model load")

        # ── train
        logger.info("Training  |  %d samples  |  epochs=%d%s",
                    len(train_df), epochs, "  max_steps=3" if test_mode else "")
        t0 = time.time()
        model.fit(train_df, conditional_col=cfg["start_col"], fd_list=cfg["fd_list"])
        report_gpu("after training")

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  [GPU peak during training] {peak:.2f} GB")

        logger.info("Training done  |  %.1f s", time.time() - t0)

    # ── sample
    logger.info("Sampling %d rows  |  max_len=%d", n_samples, cfg["max_length"])
    t0 = time.time()
    start_dist = train_df[cfg["start_col"]].value_counts(normalize=True).to_dict()
    synthetic_df = model.sample(
        n_samples=n_samples,
        start_col=cfg["start_col"],
        start_col_dist=start_dist,
        temperature=TEMPERATURE,
        k=sample_k,
        top_p=TOP_P,
        max_length=cfg["max_length"],
        drop_nan=False,
        device=DEVICE,
    )
    report_gpu("after sampling")
    logger.info("Sampling done  |  %.1f s  |  %d / %d rows",
                time.time() - t0, len(synthetic_df), n_samples)

    synthetic_df.to_csv(output_path, index=False)
    logger.info("Saved: %s", output_path)

    if not test_mode:
        compare_stats(train_df, synthetic_df, logger)

    # ── cleanup GPU memory before next dataset
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    report_gpu("after cleanup")

    logger.info("=" * 60)
    logger.info("Done: %s", name)
    logger.info("=" * 60)


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets", nargs="*",
        help="Dataset name(s) to run. Omit to run all.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: 3 training steps + 5 sample rows per dataset, checks GPU memory.",
    )
    parser.add_argument(
        "--sample-only", action="store_true",
        help="Skip training; load the latest checkpoint and sample directly.",
    )
    args = parser.parse_args()

    if args.test and args.sample_only:
        print("--test and --sample-only are mutually exclusive.")
        sys.exit(1)

    targets = args.datasets if args.datasets else list(DATASETS.keys())
    invalid = [d for d in targets if d not in DATASETS]
    if invalid:
        print(f"Unknown dataset(s): {invalid}. Available: {list(DATASETS.keys())}")
        sys.exit(1)

    mode = "TEST" if args.test else ("SAMPLE-ONLY" if args.sample_only else "FULL")
    print(f"[{mode}] Running datasets: {targets}")
    for name in targets:
        try:
            run_dataset(name, DATASETS[name], test_mode=args.test, sample_only=args.sample_only)
        except Exception as e:
            logging.getLogger(name).exception("FAILED: %s — %s", name, e)
            print(f"\n[ERROR] {name} failed: {e}\n")


if __name__ == "__main__":
    main()

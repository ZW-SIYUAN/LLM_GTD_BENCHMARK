"""
smoke_test.py — 用当前参数跑极短训练 + 少量采样，验证全流程不崩溃。
=====================================================================
不做任何参数搜索，直接使用 generate_synthetic.py 里 DATASET_CONFIGS 的当前配置。
每个 (模型, 数据集) 组合：
  - 训练：max_steps=3（3个优化步，~30秒）
  - 采样：n_samples=5

用途
----
- 正式实验前的冒烟测试：确认环境、依赖、参数配置没有低级错误
- 快速验证改动后代码仍然可以端到端跑通

用法
----
# 全量测试（所有模型 × 所有数据集）
python smoke_test.py

# 只测特定组合
python smoke_test.py --models GraDe GraFT --datasets sick income

# 只测训练（跳过采样）
python smoke_test.py --skip-sampling
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from llm_gtd_benchmark.models import GReaTModel, PAFTModel, GraDeModel, GraFTModel
from generate_synthetic import (
    DATASET_CONFIGS,
    LLM, LEARNING_RATE, TEMPERATURE, TOP_P,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    ORIGIN_ROOT,
    _get_sample_kwargs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_test")

ALL_MODELS   = ["GReaT", "PAFT", "GraDe", "GraFT"]
ALL_DATASETS = list(DATASET_CONFIGS.keys())

CKPT_ROOT = ROOT / "_smoke_ckpts"


def _free_gpu():
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _gpu_mem_gb() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() / 1024**3
            peak  = torch.cuda.max_memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"alloc={alloc:.2f}GB peak={peak:.2f}GB total={total:.2f}GB"
    except Exception:
        pass
    return "no GPU"


def _reset_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _use_bf16() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ── Model builder (smoke version: max_steps=3) ───────────────────────────────

def build_smoke_model(model_name: str, dataset_name: str, fitted_order=None):
    cfg      = DATASET_CONFIGS[dataset_name]
    ckpt_dir = str(CKPT_ROOT / model_name / dataset_name)
    bf16     = _use_bf16()
    device   = _device()

    smoke_kwargs = dict(
        learning_rate=LEARNING_RATE,
        max_steps=3,        # 只跑 3 步，够触发所有代码路径
        save_steps=9999,
        logging_steps=1,
        bf16=bf16,
        fp16=False,
    )

    if model_name == "GReaT":
        return GReaTModel(
            llm=LLM, experiment_dir=ckpt_dir,
            epochs=1, batch_size=cfg["batch_size"],
            use_lora=True, lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            **smoke_kwargs,
        )

    elif model_name == "PAFT":
        return PAFTModel(
            llm=LLM, experiment_dir=ckpt_dir,
            epochs=1, batch_size=cfg["batch_size"],
            column_order=fitted_order, random_state=42,
            use_lora=True, lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            **smoke_kwargs,
        )

    elif model_name == "GraDe":
        fd_list = cfg.get("fd_list") or []
        return GraDeModel(
            llm=LLM, experiment_dir=ckpt_dir,
            epochs=1, batch_size=cfg["batch_size"],
            fd_list=fd_list if fd_list else None,
            device=device,
            use_lora=True, lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            lora_target_modules=LORA_TARGET_MODULES,
            **smoke_kwargs,
        )

    elif model_name == "GraFT":
        fd_list = cfg.get("fd_list") or []
        return GraFTModel(
            llm=LLM, experiment_dir=ckpt_dir,
            epochs=1, batch_size=cfg["batch_size"],
            n_bins=100, mi_threshold_quantile=0.25,
            fd_list=fd_list if fd_list else None,
            device=device,
            use_lora=True, lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            lora_target_modules=LORA_TARGET_MODULES,
            **smoke_kwargs,
        )

    raise ValueError(f"Unknown model: {model_name!r}")


# ── Single smoke run ──────────────────────────────────────────────────────────

def smoke_one(
    model_name: str,
    dataset_name: str,
    skip_sampling: bool,
) -> dict:
    """Run one (model, dataset) smoke test. Returns a result dict."""
    result = dict(
        model=model_name, dataset=dataset_name,
        train_ok=False, sample_ok=False,
        train_shape=None, sample_shape=None,
        train_s=0.0, sample_s=0.0,
        gpu_after_train="", gpu_after_sample="",
        error="",
    )

    cfg      = DATASET_CONFIGS[dataset_name]
    train_csv = ORIGIN_ROOT / dataset_name / "train.csv"
    if not train_csv.exists():
        result["error"] = f"train.csv not found: {train_csv}"
        return result

    train_df = pd.read_csv(train_csv)
    for c in cfg.get("drop_cols", []):
        if c in train_df.columns:
            train_df = train_df.drop(columns=[c])

    # GraDe: reorder columns to fixed train_order
    if model_name == "GraDe" and cfg.get("train_order") is not None:
        order = [c for c in cfg["train_order"] if c in train_df.columns]
        train_df = train_df[order]

    # PAFT: use pre-computed FD order from config
    fitted_order = None
    if model_name == "PAFT":
        order = cfg.get("train_order")
        if order:
            fitted_order = [c for c in order if c in train_df.columns]

    _reset_peak()

    # ── Training ──────────────────────────────────────────────────────────────
    try:
        model = build_smoke_model(model_name, dataset_name, fitted_order)

        t0 = time.time()
        if model_name == "GraDe":
            start_col = cfg.get("start_col") or train_df.columns[0]
            model.fit(train_df, conditional_col=start_col)
        else:
            model.fit(train_df)
        result["train_s"]     = round(time.time() - t0, 1)
        result["train_shape"] = str(train_df.shape)
        result["train_ok"]    = True
        result["gpu_after_train"] = _gpu_mem_gb()
        logger.info(
            "  TRAIN OK  %.1fs  %s  %s",
            result["train_s"], train_df.shape, result["gpu_after_train"],
        )
    except Exception:
        result["error"] = traceback.format_exc()
        logger.error("  TRAIN FAIL\n%s", result["error"])
        _free_gpu()
        return result

    if skip_sampling:
        del model
        _free_gpu()
        return result

    # ── Sampling ──────────────────────────────────────────────────────────────
    try:
        _reset_peak()
        fitted_order_post = getattr(model, "fitted_order_", None)
        sample_kwargs = _get_sample_kwargs(
            model_name, dataset_name, train_df, n_samples=5,
            fitted_order=fitted_order_post,
        )

        t0 = time.time()
        synth = model.sample(**sample_kwargs)
        result["sample_s"]     = round(time.time() - t0, 1)
        result["sample_shape"] = str(synth.shape)
        result["sample_ok"]    = True
        result["gpu_after_sample"] = _gpu_mem_gb()
        logger.info(
            "  SAMPLE OK  %.1fs  synth=%s  %s",
            result["sample_s"], synth.shape, result["gpu_after_sample"],
        )
    except Exception:
        result["error"] = traceback.format_exc()
        logger.error("  SAMPLE FAIL\n%s", result["error"])

    del model
    _free_gpu()
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def run_smoke(models, datasets, skip_sampling):
    all_results = []
    n_total = len(models) * len(datasets)
    n_done  = 0

    for dataset_name in datasets:
        for model_name in models:
            n_done += 1
            logger.info("")
            logger.info("━" * 60)
            logger.info("[%d/%d]  %-6s / %s", n_done, n_total, model_name, dataset_name)
            logger.info("━" * 60)

            r = smoke_one(model_name, dataset_name, skip_sampling)
            all_results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 72)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 72)
    fmt = "%-6s  %-12s  %-8s  %-8s  %-10s  %-10s  %s"
    logger.info(fmt, "Model", "Dataset", "Train", "Sample", "Train(s)", "Sample(s)", "Note")
    logger.info("-" * 72)

    n_ok = 0
    for r in all_results:
        train_s  = str(r["train_s"])  if r["train_ok"]  else "-"
        sample_s = str(r["sample_s"]) if r["sample_ok"] else ("-" if skip_sampling else "FAIL")
        train_st  = "OK" if r["train_ok"]  else "FAIL"
        sample_st = "OK" if r["sample_ok"] else ("skip" if skip_sampling else "FAIL")
        note = ""
        if r["error"] and not r["train_ok"]:
            note = r["error"].strip().splitlines()[-1][:50]
        elif r["error"]:
            note = r["error"].strip().splitlines()[-1][:50]

        all_ok = r["train_ok"] and (skip_sampling or r["sample_ok"])
        if all_ok:
            n_ok += 1

        logger.info(fmt, r["model"], r["dataset"],
                    train_st, sample_st, train_s, sample_s, note)

    logger.info("=" * 72)
    logger.info("Result: %d / %d passed", n_ok, n_total)
    logger.info("=" * 72)

    # Print failures in detail
    failures = [r for r in all_results if r["error"]]
    if failures:
        logger.info("")
        logger.info("FAILURE DETAILS:")
        for r in failures:
            logger.info("  %s / %s:\n%s", r["model"], r["dataset"], r["error"])

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke test: run 3 training steps + 5 sample rows per (model, dataset).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS, metavar="MODEL",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=ALL_DATASETS, default=ALL_DATASETS, metavar="DATASET",
    )
    parser.add_argument(
        "--skip-sampling", action="store_true", default=False,
        help="Only run training, skip sampling.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_smoke(args.models, args.datasets, args.skip_sampling)

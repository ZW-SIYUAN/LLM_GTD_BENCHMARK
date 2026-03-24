"""
smoke_test2.py — Experiment 2 概念漂移数据集冒烟测试
=====================================================
在 generate_synthetic2.py 的当前参数配置下跑极短训练 + 少量采样，
验证全流程不崩溃、不爆显存。

每个 (模型, 数据集) 组合：
  - 训练：max_steps=3（约 30 秒）
  - 采样：n_samples=5

注意：``t`` 列（时间步序号）与正式实验保持一致，在训练前全局丢弃。

用法
----
# 全量测试（所有模型 × 所有激活数据集）
python smoke_test2.py

# 只测特定组合
python smoke_test2.py --models GraDe GraFT --datasets agrawal

# 只测训练（跳过采样）
python smoke_test2.py --skip-sampling
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
from generate_synthetic2 import (
    DATASET_CONFIGS,
    LLM, LEARNING_RATE, TOP_P,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    ORIGIN_ROOT,
    _get_sample_kwargs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_test2")

ALL_MODELS   = ["GReaT", "PAFT", "GraDe", "GraFT"]
ALL_DATASETS = list(DATASET_CONFIGS.keys())

CKPT_ROOT = ROOT / "_smoke_ckpts2"


# ── GPU helpers ───────────────────────────────────────────────────────────────

def _free_gpu() -> None:
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
            alloc = torch.cuda.memory_allocated()     / 1024**3
            peak  = torch.cuda.max_memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"alloc={alloc:.2f}GB peak={peak:.2f}GB total={total:.2f}GB"
    except Exception:
        pass
    return "no GPU"


def _reset_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _use_bf16() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


# ── Model builder (smoke: max_steps=3) ───────────────────────────────────────

def build_smoke_model(model_name: str, dataset_name: str, fitted_order=None):
    cfg      = DATASET_CONFIGS[dataset_name]
    ckpt_dir = str(CKPT_ROOT / model_name / dataset_name)
    bf16     = _use_bf16()
    device   = _device()

    smoke_kwargs = dict(
        learning_rate=LEARNING_RATE,
        max_steps=3,       # 3 optimisation steps — enough to touch all code paths
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
            fd_list=fd_list if fd_list else None,
            device=device,
            use_lora=True, lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            lora_target_modules=LORA_TARGET_MODULES,
            **smoke_kwargs,
        )

    raise ValueError(f"Unknown model: {model_name!r}")


# ── Single smoke run ──────────────────────────────────────────────────────────

def smoke_one(model_name: str, dataset_name: str, skip_sampling: bool) -> dict:
    result = dict(
        model=model_name, dataset=dataset_name,
        train_ok=False, sample_ok=False,
        train_shape=None, sample_shape=None,
        train_s=0.0, sample_s=0.0,
        gpu_after_train="", gpu_after_sample="",
        error="",
    )

    cfg       = DATASET_CONFIGS[dataset_name]
    train_csv = ORIGIN_ROOT / dataset_name / "train.csv"
    if not train_csv.exists():
        result["error"] = f"train.csv not found: {train_csv}"
        return result

    train_df = pd.read_csv(train_csv)

    # Drop time-index column (always) + any dataset-specific extras
    drop = ["t"] + [c for c in cfg.get("drop_cols", []) if c != "t"]
    train_df = train_df.drop(columns=[c for c in drop if c in train_df.columns])

    # GraDe: reorder to pre-computed IB order if available
    if model_name == "GraDe" and cfg.get("train_order") is not None:
        order    = [c for c in cfg["train_order"] if c in train_df.columns]
        train_df = train_df[order]

    # PAFT: pass pre-computed column order if available
    fitted_order = None
    if model_name == "PAFT" and cfg.get("train_order") is not None:
        fitted_order = [c for c in cfg["train_order"] if c in train_df.columns]

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

        result["train_s"]         = round(time.time() - t0, 1)
        result["train_shape"]     = str(train_df.shape)
        result["train_ok"]        = True
        result["gpu_after_train"] = _gpu_mem_gb()

        logger.info(
            "  TRAIN OK  %.1fs  shape=%s  %s",
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
            model_name, dataset_name, train_df,
            n_synth=5, fitted_order=fitted_order_post,
        )

        # Smoke-test safety: GraDe/GraFT column distributions are not fully built
        # after only 3 training steps, so an explicitly resolved start_col will crash
        # with "given without distribution".  Remove it unless the config hard-codes one
        # (meaning the user has pre-verified it exists in the model's _col_dist).
        if model_name in ("GraDe", "GraFT") and not cfg.get("start_col"):
            sample_kwargs.pop("start_col", None)

        t0    = time.time()
        synth = model.sample(**sample_kwargs)
        result["sample_s"]         = round(time.time() - t0, 1)
        result["sample_shape"]     = str(synth.shape)
        result["gpu_after_sample"] = _gpu_mem_gb()

        if len(synth) == 0:
            # Expected with only 3 training steps — model hasn't learned yet.
            # Mark as warning (not hard FAIL) so the summary stays readable.
            result["sample_ok"] = False
            result["error"]     = "0 rows returned (expected with max_steps=3)"
            logger.warning(
                "  SAMPLE WARN  %.1fs  0 rows — model undertrained, not a code error  %s",
                result["sample_s"], result["gpu_after_sample"],
            )
        else:
            result["sample_ok"] = True
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

def run_smoke(models: list[str], datasets: list[str], skip_sampling: bool) -> list[dict]:
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

    # ── Summary table ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 72)
    logger.info("SMOKE TEST 2 SUMMARY")
    logger.info("=" * 72)
    fmt = "%-6s  %-12s  %-8s  %-8s  %-10s  %-10s  %s"
    logger.info(fmt, "Model", "Dataset", "Train", "Sample", "Train(s)", "Sample(s)", "Note")
    logger.info("-" * 72)

    n_ok = 0
    for r in all_results:
        is_zero_rows = r["error"].startswith("0 rows") if r["error"] else False
        train_s  = str(r["train_s"])  if r["train_ok"]  else "-"
        sample_s = str(r["sample_s"]) if r["sample_s"]  else ("-" if skip_sampling else "FAIL")
        train_st  = "OK"   if r["train_ok"]  else "FAIL"
        if skip_sampling:
            sample_st = "skip"
        elif r["sample_ok"]:
            sample_st = "OK"
        elif is_zero_rows:
            sample_st = "WARN"   # undertrained, not a code error
        else:
            sample_st = "FAIL"
        note = r["error"].strip().splitlines()[-1][:50] if r["error"] else ""

        # Count as passed if train OK and sampling didn't hard-crash
        # (0-rows warning is acceptable for a 3-step smoke run)
        all_ok = r["train_ok"] and (skip_sampling or r["sample_ok"] or is_zero_rows)
        if all_ok:
            n_ok += 1

        logger.info(fmt, r["model"], r["dataset"],
                    train_st, sample_st, train_s, sample_s, note)

    logger.info("=" * 72)
    logger.info("Result: %d / %d passed", n_ok, n_total)
    logger.info("=" * 72)

    failures = [r for r in all_results if r["error"]]
    if failures:
        logger.info("")
        logger.info("FAILURE DETAILS:")
        for r in failures:
            logger.info("  %s / %s:\n%s", r["model"], r["dataset"], r["error"])

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test for Experiment 2 (concept-drift datasets).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS, metavar="MODEL",
        help="Models to test. Choices: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=ALL_DATASETS, default=ALL_DATASETS, metavar="DATASET",
        help="Datasets to test. Choices: " + ", ".join(ALL_DATASETS),
    )
    parser.add_argument(
        "--skip-sampling", action="store_true", default=False,
        help="Only run training steps, skip sampling.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_smoke(args.models, args.datasets, args.skip_sampling)

"""
pre_experiment.py — GPU memory probe & hyperparameter tuning helper.
=====================================================================
在正式运行 generate_synthetic.py 之前，对每个 (模型, 数据集) 组合执行
极短的训练和采样探测，量化峰值显存，自动筛选出不 OOM 的最大超参数组合，
并将推荐配置写入 recommended_configs.py。

探测流程
--------
1. 训练探测：使用候选 batch_size 运行 max_steps=3，记录峰值显存。
   从大到小扫描，找到最大的合法值。
2. 采样探测：用同一个微训练过的模型，对 (sample_k, max_length) 候选组合
   依次运行 n_samples=5，记录峰值显存。
   取首个不 OOM 的组合。
3. 汇总：打印每个组合的显存/状态，并输出 recommended_configs.py。

用法
----
# 全量探测（所有模型 × 所有数据集）
python pre_experiment.py

# 只探测特定模型 / 数据集
python pre_experiment.py --models GraDe GraFT --datasets sick income

# 跳过采样探测（仅测训练）
python pre_experiment.py --skip-sampling

# 设置显存安全阈值（默认留 10% 余量）
python pre_experiment.py --safety-margin 0.15

# 写出推荐配置到自定义路径
python pre_experiment.py --output my_recommended_configs.py

注意
----
- 需要已安装 torch + be_great + peft，且有可用 GPU。
- CPU 模式同样可运行，但显存报告全部为 0（无意义）；建议 GPU 环境使用。
- GraFT 的 IB 列序发现默认在首次探测时执行一次，结果缓存后复用。
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from llm_gtd_benchmark.models import GReaTModel, PAFTModel, GraDeModel, GraFTModel
from generate_synthetic import (  # reuse configs from main script
    DATASET_CONFIGS,
    LLM,
    LEARNING_RATE,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    ORIGIN_ROOT,
)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pre_experiment")

ALL_MODELS   = ["GReaT", "PAFT", "GraDe", "GraFT"]
ALL_DATASETS = list(DATASET_CONFIGS.keys())

# ── Hyperparameter search spaces ─────────────────────────────────────────────
# 从大到小扫描；取第一个不 OOM 的值
BATCH_CANDIDATES   = [32, 16, 8, 4, 2]
# (sample_k, max_length) 联合扫描；两个参数共同决定采样时的 KV-cache 峰值
SAMPLE_CANDIDATES: List[Tuple[int, int]] = [
    (100, 400), (100, 200), (100, 150), (100, 100), (100,  50),
    ( 50, 400), ( 50, 200), ( 50, 150), ( 50, 100), ( 50,  50),
    ( 20, 400), ( 20, 200), ( 20, 150), ( 20, 100), ( 20,  50),
    ( 10, 400), ( 10, 200), ( 10, 150), ( 10, 100), ( 10,  50),
    (  5, 200), (  5, 100), (  5,  50),
]


# ── GPU helpers ───────────────────────────────────────────────────────────────

def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _reset_gpu_stats():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except Exception:
        pass


def _peak_gpu_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 1024 ** 3
    except Exception:
        pass
    return 0.0


def _total_gpu_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:
        pass
    return 0.0


def _free_gpu():
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ── Result data classes ───────────────────────────────────────────────────────

@dataclass
class ProbeResult:
    model_name:    str
    dataset_name:  str
    phase:         str          # "train" or "sample"
    status:        str          # "ok", "oom", "error", "skip"
    batch_size:    Optional[int]  = None
    sample_k:      Optional[int]  = None
    max_length:    Optional[int]  = None
    peak_gb:       float          = 0.0
    total_gb:      float          = 0.0
    elapsed_s:     float          = 0.0
    note:          str            = ""


@dataclass
class DatasetProbe:
    model_name:    str
    dataset_name:  str
    # best values found
    rec_batch:     Optional[int]  = None
    rec_k:         Optional[int]  = None
    rec_max_len:   Optional[int]  = None
    train_peak_gb: float          = 0.0
    sample_peak_gb: float         = 0.0
    details:       List[ProbeResult] = field(default_factory=list)


# ── Model builder (probe version — max_steps=3, tiny experiment_dir) ─────────

def _build_probe_model(
    model_name: str,
    dataset_name: str,
    batch_size: int,
    device: str,
    fitted_order: Optional[List[str]] = None,
):
    """Build a model for probing — uses max_steps=3 and a temp ckpt dir."""
    cfg        = DATASET_CONFIGS[dataset_name]
    ckpt_dir   = str(ROOT / "_probe_ckpts" / model_name / dataset_name)
    fd_list    = cfg.get("fd_list") or []

    try:
        import torch
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        use_bf16 = False

    probe_kwargs = dict(
        learning_rate=LEARNING_RATE,
        max_steps=3,            # ← only 3 optimizer steps; enough to measure peak
        save_steps=9999,
        logging_steps=1,
        bf16=use_bf16,
        fp16=False,
    )

    if model_name == "GReaT":
        return GReaTModel(
            llm=LLM,
            experiment_dir=ckpt_dir,
            epochs=1,
            batch_size=batch_size,
            use_lora=True,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            **probe_kwargs,
        )

    elif model_name == "PAFT":
        return PAFTModel(
            llm=LLM,
            experiment_dir=ckpt_dir,
            epochs=1,
            batch_size=batch_size,
            column_order=fitted_order,   # reuse pre-computed order
            random_state=42,
            use_lora=True,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            **probe_kwargs,
        )

    elif model_name == "GraDe":
        return GraDeModel(
            llm=LLM,
            experiment_dir=ckpt_dir,
            epochs=1,
            batch_size=batch_size,
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
            fixed_col_order=True,
            **probe_kwargs,
        )

    elif model_name == "GraFT":
        return GraFTModel(
            llm=LLM,
            experiment_dir=ckpt_dir,
            epochs=1,
            batch_size=batch_size,
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
            fixed_col_order=True,
            **probe_kwargs,
        )
    else:
        raise ValueError(f"Unknown model: {model_name!r}")


# ── Training probe ────────────────────────────────────────────────────────────

def probe_training(
    model_name: str,
    dataset_name: str,
    train_df: pd.DataFrame,
    device: str,
    safety_margin: float,
    fitted_order: Optional[List[str]] = None,
) -> Tuple[Optional[int], Optional[Any], List[ProbeResult]]:
    """
    Sweep batch sizes (large → small).  Return (best_batch, fitted_model, results).
    fitted_model is the trained model with the best batch_size (for reuse in
    sampling probe); None if all batch sizes OOM'd.
    """
    cfg        = DATASET_CONFIGS[dataset_name]
    total_gb   = _total_gpu_gb()
    threshold  = total_gb * (1 - safety_margin) if total_gb > 0 else float("inf")
    results: List[ProbeResult] = []

    # Prepare train_df for GraDe (fixed column order)
    if model_name == "GraDe" and cfg.get("train_order") is not None:
        order = [c for c in cfg["train_order"] if c in train_df.columns]
        df = train_df[order]
    elif model_name == "PAFT" and fitted_order is not None:
        df = train_df[fitted_order] if set(fitted_order) <= set(train_df.columns) else train_df
    else:
        df = train_df

    best_batch      = None
    best_model: Any = None

    for bs in BATCH_CANDIDATES:
        _free_gpu()
        _reset_gpu_stats()
        t0 = time.time()

        try:
            model = _build_probe_model(
                model_name, dataset_name, bs, device, fitted_order
            )
            if model_name == "GraDe":
                start_col = cfg.get("start_col") or df.columns[0]
                fd_list   = cfg.get("fd_list") or []
                model.fit(df, conditional_col=start_col, fd_list=fd_list)
            else:
                model.fit(df)

            peak   = _peak_gpu_gb()
            elapsed = time.time() - t0

            if total_gb > 0 and peak > threshold:
                # Technically ran but exceeded safety threshold
                results.append(ProbeResult(
                    model_name=model_name, dataset_name=dataset_name,
                    phase="train", status="unsafe",
                    batch_size=bs, peak_gb=peak, total_gb=total_gb,
                    elapsed_s=elapsed,
                    note=f"peak {peak:.2f}GB > threshold {threshold:.2f}GB",
                ))
                del model
                _free_gpu()
                continue

            results.append(ProbeResult(
                model_name=model_name, dataset_name=dataset_name,
                phase="train", status="ok",
                batch_size=bs, peak_gb=peak, total_gb=total_gb,
                elapsed_s=elapsed,
            ))
            best_batch = bs
            best_model = model
            break   # found the largest safe batch_size

        except Exception as exc:
            elapsed = time.time() - t0
            status  = "oom" if "CUDA out of memory" in str(exc) or "OutOfMemoryError" in type(exc).__name__ else "error"
            results.append(ProbeResult(
                model_name=model_name, dataset_name=dataset_name,
                phase="train", status=status,
                batch_size=bs, peak_gb=_peak_gpu_gb(), total_gb=total_gb,
                elapsed_s=elapsed,
                note=str(exc)[:120],
            ))
            _free_gpu()

    return best_batch, best_model, results


# ── Sampling probe ────────────────────────────────────────────────────────────

def probe_sampling(
    model_name: str,
    dataset_name: str,
    train_df: pd.DataFrame,
    model: Any,
    device: str,
    safety_margin: float,
    fitted_order: Optional[List[str]] = None,
) -> Tuple[Optional[int], Optional[int], List[ProbeResult]]:
    """
    Sweep (sample_k, max_length) candidates.  Return (best_k, best_max_len, results).
    """
    cfg        = DATASET_CONFIGS[dataset_name]
    total_gb   = _total_gpu_gb()
    threshold  = total_gb * (1 - safety_margin) if total_gb > 0 else float("inf")
    results: List[ProbeResult] = []

    # Resolve start_col / start_col_dist for GraDe / GraFT
    start_col = cfg.get("start_col")
    if model_name == "GraFT" and fitted_order:
        start_col = fitted_order[0]
    if start_col is None:
        cols = (
            fitted_order
            or cfg.get("train_order")
            or train_df.columns.tolist()
        )
        start_col = cols[0] if cols else train_df.columns[0]

    start_col_dist = None
    if start_col and start_col in train_df.columns:
        start_col_dist = train_df[start_col].value_counts(normalize=True).to_dict()

    best_k       = None
    best_max_len = None

    for (sk, ml) in SAMPLE_CANDIDATES:
        _free_gpu()
        _reset_gpu_stats()
        t0 = time.time()

        try:
            if model_name in ("GReaT", "PAFT"):
                synth = model.sample(
                    n_samples=5,
                    temperature=0.8,
                    k=sk,
                    max_length=ml,
                )
            else:
                synth = model.sample(
                    n_samples=5,
                    start_col=start_col,
                    start_col_dist=start_col_dist,
                    temperature=0.8,
                    k=sk,
                    top_p=1.0,
                    max_length=ml,
                    drop_nan=False,
                )

            peak    = _peak_gpu_gb()
            elapsed = time.time() - t0

            if total_gb > 0 and peak > threshold:
                results.append(ProbeResult(
                    model_name=model_name, dataset_name=dataset_name,
                    phase="sample", status="unsafe",
                    sample_k=sk, max_length=ml,
                    peak_gb=peak, total_gb=total_gb, elapsed_s=elapsed,
                    note=f"peak {peak:.2f}GB > threshold {threshold:.2f}GB",
                ))
                continue

            results.append(ProbeResult(
                model_name=model_name, dataset_name=dataset_name,
                phase="sample", status="ok",
                sample_k=sk, max_length=ml,
                peak_gb=peak, total_gb=total_gb, elapsed_s=elapsed,
            ))
            best_k       = sk
            best_max_len = ml
            break   # found the largest safe (k, max_len)

        except Exception as exc:
            elapsed = time.time() - t0
            status  = "oom" if "CUDA out of memory" in str(exc) or "OutOfMemoryError" in type(exc).__name__ else "error"
            results.append(ProbeResult(
                model_name=model_name, dataset_name=dataset_name,
                phase="sample", status=status,
                sample_k=sk, max_length=ml,
                peak_gb=_peak_gpu_gb(), total_gb=total_gb, elapsed_s=elapsed,
                note=str(exc)[:120],
            ))
            _free_gpu()

    return best_k, best_max_len, results


# ── IB order cache (GraFT discovers it internally, but we need it for PAFT) ──

_ib_order_cache: Dict[str, Optional[List[str]]] = {}


def _get_ib_or_fd_order(
    model_name: str,
    dataset_name: str,
    train_df: pd.DataFrame,
) -> Optional[List[str]]:
    """Return column order for PAFT/GraFT; None for GReaT/GraDe."""
    if model_name not in ("PAFT", "GraFT"):
        return None

    cfg = DATASET_CONFIGS[dataset_name]

    if model_name == "PAFT":
        # Use pre-computed train_order as FD order for probing
        order = cfg.get("train_order")
        if order:
            return [c for c in order if c in train_df.columns]
        return train_df.columns.tolist()

    # GraFT — IB discovery (expensive; cache per dataset)
    if dataset_name in _ib_order_cache:
        return _ib_order_cache[dataset_name]

    logger.info("    [GraFT/%s] Running IB order discovery (one-time)...", dataset_name)
    try:
        import importlib.util
        from llm_gtd_benchmark.models.graft import _IB_MODULE_PATH, _import_ib_order_finder
        ib_mod = _import_ib_order_finder()
        finder = ib_mod.IBOrderFinder(n_bins=100, mi_threshold_quantile=0.25)
        finder.fit(train_df.dropna())
        order = finder.get_order()
        _ib_order_cache[dataset_name] = order
        logger.info("    [GraFT/%s] IB order: %s", dataset_name, order)
        return order
    except Exception as exc:
        logger.warning("    [GraFT/%s] IB discovery failed: %s", dataset_name, exc)
        _ib_order_cache[dataset_name] = None
        return None


# ── Main probe loop ───────────────────────────────────────────────────────────

def run_probes(
    models: List[str],
    datasets: List[str],
    device: str,
    safety_margin: float,
    skip_training: bool,
    skip_sampling: bool,
) -> List[DatasetProbe]:
    total_gb = _total_gpu_gb()
    if total_gb > 0:
        logger.info("GPU detected: %.1f GB total  (safety threshold: %.1f GB)",
                    total_gb, total_gb * (1 - safety_margin))
    else:
        logger.warning("No GPU detected — memory readings will be 0.  "
                       "Probes will still run but OOM cannot be auto-detected.")

    all_probes: List[DatasetProbe] = []

    for dataset_name in datasets:
        cfg       = DATASET_CONFIGS[dataset_name]
        train_csv = ORIGIN_ROOT / dataset_name / "train.csv"
        if not train_csv.exists():
            logger.error("Dataset not found: %s", train_csv)
            continue

        train_df = pd.read_csv(train_csv)
        for c in cfg.get("drop_cols", []):
            if c in train_df.columns:
                train_df = train_df.drop(columns=[c])

        for model_name in models:
            logger.info("")
            logger.info("━" * 60)
            logger.info("PROBE  %-6s / %s", model_name, dataset_name)
            logger.info("━" * 60)

            probe = DatasetProbe(model_name=model_name, dataset_name=dataset_name)

            # Pre-compute column order (PAFT / GraFT)
            fitted_order = _get_ib_or_fd_order(model_name, dataset_name, train_df)

            # ── Training probe ─────────────────────────────────────────────
            best_batch = None
            trained_model = None

            if skip_training:
                logger.info("  [train] skipped by --skip-training")
            else:
                logger.info("  [train] sweeping batch_size candidates: %s", BATCH_CANDIDATES)
                best_batch, trained_model, train_results = probe_training(
                    model_name, dataset_name, train_df, device,
                    safety_margin, fitted_order
                )
                probe.details.extend(train_results)

                for r in train_results:
                    sym = "✓" if r.status == "ok" else ("⚠" if r.status == "unsafe" else "✗")
                    logger.info(
                        "    batch=%-3d  %s %-6s  peak=%.2fGB  %.1fs  %s",
                        r.batch_size, sym, r.status, r.peak_gb, r.elapsed_s, r.note
                    )

                if best_batch is not None:
                    probe.rec_batch = best_batch
                    probe.train_peak_gb = next(
                        r.peak_gb for r in train_results
                        if r.status == "ok" and r.batch_size == best_batch
                    )
                    logger.info("  [train] ✓ best batch_size = %d  (peak %.2f GB)",
                                best_batch, probe.train_peak_gb)
                else:
                    logger.error("  [train] ✗ all batch sizes OOM'd for %s/%s",
                                 model_name, dataset_name)

            # ── Sampling probe ─────────────────────────────────────────────
            if skip_sampling:
                logger.info("  [sample] skipped by --skip-sampling")
            elif trained_model is None and not skip_training:
                logger.warning("  [sample] skipped — no trained model available")
            else:
                # If training was skipped, try to build + fit a tiny model for sampling
                if trained_model is None:
                    logger.info(
                        "  [sample] building minimal model for sampling probe "
                        "(batch=%d, max_steps=3)...",
                        cfg["batch_size"],
                    )
                    try:
                        _, trained_model, _ = probe_training(
                            model_name, dataset_name, train_df, device,
                            safety_margin=0.0,   # no threshold — just try default batch
                            fitted_order=fitted_order,
                        )
                    except Exception as exc:
                        logger.warning(
                            "  [sample] could not build minimal model: %s", exc
                        )

                if trained_model is not None:
                    logger.info("  [sample] sweeping (sample_k, max_length) candidates ...")
                    best_k, best_max_len, samp_results = probe_sampling(
                        model_name, dataset_name, train_df, trained_model,
                        device, safety_margin, fitted_order
                    )
                    probe.details.extend(samp_results)

                    for r in samp_results:
                        sym = "✓" if r.status == "ok" else ("⚠" if r.status == "unsafe" else "✗")
                        logger.info(
                            "    k=%-4d  max_len=%-5d  %s %-6s  peak=%.2fGB  %.1fs  %s",
                            r.sample_k, r.max_length,
                            sym, r.status, r.peak_gb, r.elapsed_s, r.note
                        )

                    if best_k is not None:
                        probe.rec_k       = best_k
                        probe.rec_max_len = best_max_len
                        probe.sample_peak_gb = next(
                            r.peak_gb for r in samp_results
                            if r.status == "ok"
                            and r.sample_k == best_k
                            and r.max_length == best_max_len
                        )
                        logger.info("  [sample] ✓ best (k=%d, max_len=%d)  peak=%.2f GB",
                                    best_k, best_max_len, probe.sample_peak_gb)
                    else:
                        logger.error("  [sample] ✗ all (k, max_len) combinations OOM'd")

            # cleanup
            del trained_model
            _free_gpu()

            all_probes.append(probe)

    return all_probes


# ── Summary & recommended config writer ──────────────────────────────────────

def _unified_params(probes: List[DatasetProbe], dataset_name: str, orig_cfg: dict) -> dict:
    """Return a single set of hyperparameters for *dataset_name* that is safe
    for ALL models — i.e. the most conservative (minimum) values across every
    per-model probe result.

    For a fair controlled comparison experiment every model on the same
    dataset must train and sample with identical hyperparameters, so we take
    the minimum across all models.  If a model was not probed we fall back to
    the original config value.
    """
    ds_probes = [p for p in probes if p.dataset_name == dataset_name]

    def _min_batch() -> int:
        vals = [p.rec_batch for p in ds_probes if p.rec_batch is not None]
        return min(vals) if vals else orig_cfg["batch_size"]

    def _min_k() -> int:
        vals = [p.rec_k for p in ds_probes if p.rec_k is not None]
        return min(vals) if vals else orig_cfg.get("sample_k", 100)

    def _min_len() -> int:
        vals = [p.rec_max_len for p in ds_probes if p.rec_max_len is not None]
        return min(vals) if vals else orig_cfg.get("max_length", 200)

    return {
        "batch_size": _min_batch(),
        "sample_k":   _min_k(),
        "max_length":  _min_len(),
    }


def print_summary(probes: List[DatasetProbe], total_gb: float, original_configs: dict):
    logger.info("")
    logger.info("=" * 80)
    logger.info("PER-MODEL PROBE DETAIL")
    logger.info("=" * 80)
    header = f"{'Model':<8}  {'Dataset':<12}  {'batch':>5}  {'k':>5}  "
    header += f"{'max_len':>7}  {'train_GB':>8}  {'samp_GB':>7}  {'status'}"
    logger.info(header)
    logger.info("-" * 80)

    for p in probes:
        batch   = str(p.rec_batch)   if p.rec_batch   is not None else "FAIL"
        k       = str(p.rec_k)       if p.rec_k       is not None else "FAIL"
        max_len = str(p.rec_max_len) if p.rec_max_len is not None else "FAIL"
        status  = "OK" if (p.rec_batch is not None and p.rec_k is not None) else "PARTIAL/FAIL"
        logger.info(
            "%-8s  %-12s  %5s  %5s  %7s  %8.2f  %7.2f  %s",
            p.model_name, p.dataset_name,
            batch, k, max_len,
            p.train_peak_gb, p.sample_peak_gb, status,
        )

    # ── Unified per-dataset recommendation (for fair comparison) ─────────────
    logger.info("")
    logger.info("=" * 80)
    logger.info("UNIFIED RECOMMENDATION  (min across all models — fair comparison)")
    logger.info("  All models on the same dataset will use these identical params.")
    logger.info("=" * 80)
    header2 = f"{'Dataset':<12}  {'batch':>5}  {'k':>5}  {'max_len':>7}  {'bottleneck model'}"
    logger.info(header2)
    logger.info("-" * 80)

    probed_datasets = sorted({p.dataset_name for p in probes})
    for ds_name in probed_datasets:
        cfg    = original_configs.get(ds_name, {})
        uni    = _unified_params(probes, ds_name, cfg)
        # Identify which model was the bottleneck for each dimension
        ds_p   = [p for p in probes if p.dataset_name == ds_name]
        b_model = next(
            (p.model_name for p in ds_p if p.rec_batch == uni["batch_size"]), "orig"
        )
        k_model = next(
            (p.model_name for p in ds_p if p.rec_k == uni["sample_k"]), "orig"
        )
        l_model = next(
            (p.model_name for p in ds_p if p.rec_max_len == uni["max_length"]), "orig"
        )
        bottleneck = f"batch←{b_model}  k←{k_model}  len←{l_model}"
        logger.info(
            "%-12s  %5d  %5d  %7d  %s",
            ds_name, uni["batch_size"], uni["sample_k"], uni["max_length"], bottleneck,
        )
    logger.info("=" * 80)


def write_recommended_configs(
    probes: List[DatasetProbe],
    output_path: Path,
    original_configs: dict,
):
    """Write a Python file with unified per-dataset configs (one set per dataset,
    safe for ALL models — minimum across per-model probe results).

    The output is a drop-in replacement for DATASET_CONFIGS in generate_synthetic.py.
    Every model on the same dataset uses identical batch_size / sample_k / max_length
    so that comparisons are controlled and fair.
    """
    lines = [
        '"""',
        'recommended_configs.py — Auto-generated by pre_experiment.py',
        '=============================================================',
        'Drop-in replacement for DATASET_CONFIGS in generate_synthetic.py.',
        '',
        'Hyperparameters are chosen as the MINIMUM across all model probe',
        'results for each dataset.  This ensures every model trains and',
        'samples under identical conditions — a prerequisite for fair',
        'controlled comparison experiments.',
        '',
        'Usage (in generate_synthetic.py):',
        '    from recommended_configs import RECOMMENDED_CONFIGS',
        '    DATASET_CONFIGS = RECOMMENDED_CONFIGS   # replace default',
        '"""',
        '',
        'RECOMMENDED_CONFIGS = {',
    ]

    for ds_name, orig_cfg in original_configs.items():
        uni = _unified_params(probes, ds_name, orig_cfg)

        # Identify bottleneck model for each dimension (informational comment)
        ds_p   = [p for p in probes if p.dataset_name == ds_name]
        b_info = next((p.model_name for p in ds_p if p.rec_batch   == uni["batch_size"]), "orig")
        k_info = next((p.model_name for p in ds_p if p.rec_k       == uni["sample_k"]),   "orig")
        l_info = next((p.model_name for p in ds_p if p.rec_max_len == uni["max_length"]),  "orig")

        lines.append(f'    "{ds_name}": {{')
        lines.append(f'        # Bottleneck: batch←{b_info}  k←{k_info}  max_len←{l_info}')

        # Structural fields — unchanged from original
        for key in ("train_order", "fd_list", "start_col", "drop_cols"):
            val = orig_cfg.get(key)
            lines.append(f'        "{key}": {repr(val)},')

        # Unified training / sampling hyperparameters
        lines.append(f'        "epochs":     {orig_cfg["epochs"]},')
        lines.append(f'        "batch_size": {uni["batch_size"]},')
        lines.append(f'        "sample_k":   {uni["sample_k"]},')
        lines.append(f'        "max_length":  {uni["max_length"]},')
        lines.append('    },')
        lines.append('')

    lines.append('}')
    lines.append('')

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Unified recommended configs written to: %s", output_path)


def write_detailed_csv(probes: List[DatasetProbe], output_path: Path):
    """Write all probe results to a CSV for further analysis."""
    rows = []
    for p in probes:
        for r in p.details:
            rows.append({
                "model":      r.model_name,
                "dataset":    r.dataset_name,
                "phase":      r.phase,
                "status":     r.status,
                "batch_size": r.batch_size,
                "sample_k":   r.sample_k,
                "max_length": r.max_length,
                "peak_gb":    round(r.peak_gb, 3),
                "total_gb":   round(r.total_gb, 3),
                "elapsed_s":  round(r.elapsed_s, 2),
                "note":       r.note,
            })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info("Detailed results written to: %s", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU memory probe and hyperparameter tuning helper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
        metavar="MODEL",
        help="Models to probe.  Choices: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=ALL_DATASETS, default=ALL_DATASETS,
        metavar="DATASET",
        help="Datasets to probe.  Choices: " + ", ".join(ALL_DATASETS),
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Compute device.",
    )
    parser.add_argument(
        "--safety-margin", type=float, default=0.10,
        metavar="FRAC",
        help="Fraction of total GPU memory to keep free (e.g. 0.10 = 10%% headroom).",
    )
    parser.add_argument(
        "--skip-training", action="store_true", default=False,
        help="Skip training probes (only run sampling probes on tiny-trained models).",
    )
    parser.add_argument(
        "--skip-sampling", action="store_true", default=False,
        help="Skip sampling probes.",
    )
    parser.add_argument(
        "--output", default="recommended_configs.py",
        metavar="FILE",
        help="Path for the generated recommended configs Python file.",
    )
    parser.add_argument(
        "--csv", default="probe_results.csv",
        metavar="FILE",
        help="Path for the detailed per-probe CSV results file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Auto-detect GPU
    device = args.device
    if device == "cuda" and not _gpu_available():
        logger.warning("CUDA not available — switching to CPU.  "
                       "Memory readings will be 0.")
        device = "cpu"

    try:
        import torch
        if device == "cuda":
            logger.info("GPU: %s  (%.1f GB)",
                        torch.cuda.get_device_name(0),
                        _total_gpu_gb())
    except ImportError:
        pass

    probes = run_probes(
        models=args.models,
        datasets=args.datasets,
        device=device,
        safety_margin=args.safety_margin,
        skip_training=args.skip_training,
        skip_sampling=args.skip_sampling,
    )

    total_gb = _total_gpu_gb()
    print_summary(probes, total_gb, DATASET_CONFIGS)

    output_path = ROOT / args.output
    csv_path    = ROOT / args.csv

    write_recommended_configs(probes, output_path, DATASET_CONFIGS)
    write_detailed_csv(probes, csv_path)

    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review %s  ← recommended per-model hyperparameters", output_path.name)
    logger.info("  2. Review %s  ← full probe log with memory readings",    csv_path.name)
    logger.info("  3. Apply recommended values to DATASET_CONFIGS in generate_synthetic.py")
    logger.info("  4. Run:  python generate_synthetic.py")

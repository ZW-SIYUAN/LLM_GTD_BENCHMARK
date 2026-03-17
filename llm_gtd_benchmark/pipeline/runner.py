"""BenchmarkPipeline — sequential multi-dimension evaluation runner."""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from llm_gtd_benchmark.core.result_bundle import (
    BundleMetadata,
    ResultBundle,
    schema_fingerprint,
)
from llm_gtd_benchmark.metrics.dimension0 import StructuralInterceptor
from llm_gtd_benchmark.metrics.dimension1 import FidelityEvaluator
from llm_gtd_benchmark.metrics.dimension2 import LogicEvaluator
from llm_gtd_benchmark.metrics.dimension3 import MLUtilityEvaluator
from llm_gtd_benchmark.metrics.dimension4 import PrivacyEvaluator
from llm_gtd_benchmark.metrics.dimension5 import FairnessEvaluator
from llm_gtd_benchmark.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

_SKIP_PREFIX = "skipped: "


class BenchmarkPipeline:
    """Run all configured benchmark dimensions and return a
    :class:`~llm_gtd_benchmark.core.result_bundle.ResultBundle`.

    Execution model
    ---------------
    - Dimension 0 always runs first.  If it **raises**, all downstream
      dimensions are marked ``"skipped: Dim0 failed"`` in
      ``ResultBundle.errors`` and the bundle is returned immediately.
    - Every other dimension is wrapped in an independent try/except block.
      A failure in Dim2 does **not** prevent Dim3, Dim4, or Dim5 from running.
    - Dimensions that are intentionally skipped due to missing configuration
      (e.g., Dim3 without ``target_col``) are also recorded in
      ``ResultBundle.errors`` with a ``"skipped: …"`` prefix so callers can
      distinguish configuration omissions from runtime failures.

    Parameters
    ----------
    config:
        :class:`~llm_gtd_benchmark.pipeline.config.PipelineConfig` specifying
        data, optional specs, and run options.

    Examples
    --------
    >>> config = PipelineConfig(
    ...     schema=schema,
    ...     train_real_df=train_df,
    ...     test_real_df=test_df,
    ...     model_name="GReaT",
    ...     dataset_name="adult",
    ...     target_col="income",
    ...     task_type=TaskType.BINARY_CLASS,
    ...     n_boot=1000,
    ... )
    >>> bundle = BenchmarkPipeline(config).run(synth_df)
    >>> bundle.save("results/great_adult.json")
    >>> if bundle.has_errors:
    ...     for dim, msg in bundle.errors.items():
    ...         print(f"{dim}: {msg[:120]}")
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, synth_df: pd.DataFrame) -> ResultBundle:
        """Evaluate *synth_df* across all configured dimensions.

        Parameters
        ----------
        synth_df:
            Raw synthetic DataFrame as produced by the generator (string-typed
            columns are acceptable — Dim0 handles type coercion).

        Returns
        -------
        :class:`~llm_gtd_benchmark.core.result_bundle.ResultBundle`
        """
        cfg = self.config
        dims = set(cfg.dimensions)
        errors: dict = {}

        metadata = BundleMetadata(
            model_name=cfg.model_name,
            dataset_name=cfg.dataset_name,
            schema_fingerprint=schema_fingerprint(cfg.schema),
            timestamp=datetime.now(timezone.utc).isoformat(),
            notes=cfg.notes,
        )
        bundle = ResultBundle(metadata=metadata)

        # ── Dim 0 (mandatory) ─────────────────────────────────────────────────
        clean_df: Optional[pd.DataFrame] = None
        if 0 in dims:
            logger.info("Pipeline [%s/%s]: running Dim0…", cfg.model_name, cfg.dataset_name)
            try:
                result0 = StructuralInterceptor(cfg.schema).evaluate(synth_df)
                bundle.result0 = result0
                clean_df = result0.clean_df
                logger.info(
                    "Pipeline: Dim0 OK — IRR=%.4f, clean=%d rows.",
                    result0.irr, result0.n_clean,
                )
            except Exception:
                tb = traceback.format_exc()
                errors["dim0"] = tb
                logger.error("Pipeline: Dim0 FAILED:\n%s", tb)
                for d in [1, 2, 3, 4, 5]:
                    if d in dims:
                        errors[f"dim{d}"] = _SKIP_PREFIX + "Dim0 failed"
                bundle.errors = errors
                return bundle

        # ── Dim 1 ─────────────────────────────────────────────────────────────
        if 1 in dims and clean_df is not None:
            logger.info("Pipeline: running Dim1…")
            try:
                bundle.result1 = FidelityEvaluator(
                    cfg.schema,
                    cfg.train_real_df,
                    random_state=cfg.random_state,
                ).evaluate(clean_df, n_boot=cfg.n_boot, boot_ci=cfg.boot_ci)
                logger.info("Pipeline: Dim1 OK.")
            except Exception:
                errors["dim1"] = traceback.format_exc()
                logger.error("Pipeline: Dim1 FAILED:\n%s", errors["dim1"])

        # ── Dim 2 ─────────────────────────────────────────────────────────────
        if 2 in dims and clean_df is not None:
            logger.info("Pipeline: running Dim2…")
            try:
                bundle.result2 = LogicEvaluator(
                    cfg.schema,
                    cfg.train_real_df,
                    logic_spec=cfg.logic_spec,
                ).evaluate(clean_df)
                logger.info("Pipeline: Dim2 OK.")
            except Exception:
                errors["dim2"] = traceback.format_exc()
                logger.error("Pipeline: Dim2 FAILED:\n%s", errors["dim2"])

        # ── Dim 3 ─────────────────────────────────────────────────────────────
        if 3 in dims and clean_df is not None:
            if cfg.target_col is None or cfg.task_type is None:
                msg = _SKIP_PREFIX + "target_col and task_type are required for Dim3"
                errors["dim3"] = msg
                logger.warning("Pipeline: Dim3 — %s.", msg)
            elif cfg.test_real_df is None:
                msg = _SKIP_PREFIX + "test_real_df is required for Dim3 (TSTR)"
                errors["dim3"] = msg
                logger.warning("Pipeline: Dim3 — %s.", msg)
            else:
                logger.info("Pipeline: running Dim3…")
                try:
                    bundle.result3 = MLUtilityEvaluator(
                        cfg.schema,
                        cfg.target_col,
                        cfg.task_type,
                        tuning_mode=cfg.dim3_tuning_mode,
                        random_state=cfg.random_state,
                    ).evaluate(clean_df, cfg.test_real_df, cfg.train_real_df)
                    logger.info("Pipeline: Dim3 OK.")
                except Exception:
                    errors["dim3"] = traceback.format_exc()
                    logger.error("Pipeline: Dim3 FAILED:\n%s", errors["dim3"])

        # ── Dim 4 ─────────────────────────────────────────────────────────────
        if 4 in dims and clean_df is not None:
            logger.info("Pipeline: running Dim4…")
            try:
                bundle.result4 = PrivacyEvaluator(
                    cfg.schema,
                    cfg.train_real_df,
                    random_state=cfg.random_state,
                ).evaluate(clean_df, n_boot=cfg.n_boot, boot_ci=cfg.boot_ci)
                logger.info("Pipeline: Dim4 OK.")
            except Exception:
                errors["dim4"] = traceback.format_exc()
                logger.error("Pipeline: Dim4 FAILED:\n%s", errors["dim4"])

        # ── Dim 5 ─────────────────────────────────────────────────────────────
        if 5 in dims and clean_df is not None:
            if cfg.fair_spec is None:
                msg = _SKIP_PREFIX + "fair_spec is required for Dim5"
                errors["dim5"] = msg
                logger.warning("Pipeline: Dim5 — %s.", msg)
            elif cfg.test_real_df is None:
                msg = _SKIP_PREFIX + "test_real_df is required for Dim5"
                errors["dim5"] = msg
                logger.warning("Pipeline: Dim5 — %s.", msg)
            else:
                logger.info("Pipeline: running Dim5…")
                try:
                    bundle.result5 = FairnessEvaluator(
                        cfg.schema,
                        cfg.fair_spec,
                        cfg.train_real_df,
                        random_state=cfg.random_state,
                    ).evaluate(
                        clean_df,
                        cfg.test_real_df,
                        n_boot=cfg.n_boot,
                        boot_ci=cfg.boot_ci,
                    )
                    logger.info("Pipeline: Dim5 OK.")
                except Exception:
                    errors["dim5"] = traceback.format_exc()
                    logger.error("Pipeline: Dim5 FAILED:\n%s", errors["dim5"])

        bundle.errors = errors
        n_ok = len(bundle.dimensions_computed)
        n_real_err = sum(
            1 for v in errors.values() if not v.startswith(_SKIP_PREFIX)
        )
        logger.info(
            "Pipeline: complete — %d dim(s) succeeded, %d runtime error(s).",
            n_ok,
            n_real_err,
        )
        return bundle

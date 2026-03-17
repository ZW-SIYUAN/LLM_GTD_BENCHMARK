"""PipelineConfig — unified experiment configuration for BenchmarkPipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from llm_gtd_benchmark.core.logic_spec import LogicSpec
from llm_gtd_benchmark.core.schema import DataSchema
from llm_gtd_benchmark.metrics.dimension3 import TaskType
from llm_gtd_benchmark.metrics.dimension5 import FairSpec


@dataclass
class PipelineConfig:
    """Configuration for :class:`~llm_gtd_benchmark.pipeline.BenchmarkPipeline`.

    Parameters
    ----------
    schema:
        :class:`~llm_gtd_benchmark.core.schema.DataSchema` built from the
        real training set.  Shared across all dimensions.
    train_real_df:
        Real training DataFrame.  Reference for Dim1/2/4/5 evaluation.
    model_name:
        Human-readable generator model identifier (stored in bundle metadata).
    dataset_name:
        Human-readable dataset identifier (stored in bundle metadata).
    test_real_df:
        Real held-out test DataFrame.  Required for Dim3 (TSTR) and Dim5
        disparity metrics.  Both dimensions are skipped with a descriptive
        message in ``ResultBundle.errors`` when ``None``.
    dimensions:
        Which dimensions to evaluate.  Defaults to ``[0, 1, 2, 3, 4, 5]``.
        Dimension 0 **must** be included (it produces ``clean_df`` for all
        other dimensions); omitting it raises :class:`ValueError`.
    target_col:
        Prediction target column name.  Required for Dim3 and Dim5.
    task_type:
        :class:`~llm_gtd_benchmark.metrics.dimension3.TaskType` enum value.
        Required for Dim3 and Dim5.
    logic_spec:
        :class:`~llm_gtd_benchmark.core.logic_spec.LogicSpec` for Dim2.
        When ``None``, Dim2 still computes DSI but skips ICVR / HCS / MDI.
    fair_spec:
        :class:`~llm_gtd_benchmark.metrics.dimension5.FairSpec` for Dim5.
        When ``None`` and 5 is in ``dimensions``, Dim5 is skipped.
    n_boot:
        Number of bootstrap replicates for per-metric CI computation
        (Dim1, Dim4, Dim5).  ``0`` disables bootstrap entirely.
        Default: ``0``.
    boot_ci:
        Confidence level for bootstrap CIs.  Default: ``0.95``.
    random_state:
        Seed for all stochastic operations across all dimensions.
        Default: ``42``.
    dim3_tuning_mode:
        Hyperparameter tuning strategy for :class:`~llm_gtd_benchmark.metrics.dimension3.MLUtilityEvaluator`.
        ``"shared"`` *(default)* — tune once on real training data, reuse the
        best parameters for both TRTR and TSTR.  Faster and eliminates
        hyperparameter choice as a confounding variable.
        ``"independent"`` — tune separately for each split (original behaviour).
    notes:
        Free-text annotation stored in :class:`~llm_gtd_benchmark.core.result_bundle.BundleMetadata`.

    Raises
    ------
    ValueError:
        If dimension 0 is absent from ``dimensions``, ``n_boot < 0``,
        ``boot_ci`` is not in ``(0, 1)``, or ``dim3_tuning_mode`` is not a
        recognised value.
    """

    schema: DataSchema
    train_real_df: pd.DataFrame
    model_name: str
    dataset_name: str

    test_real_df: Optional[pd.DataFrame] = None
    dimensions: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    target_col: Optional[str] = None
    task_type: Optional[TaskType] = None
    logic_spec: Optional[LogicSpec] = None
    fair_spec: Optional[FairSpec] = None
    n_boot: int = 0
    boot_ci: float = 0.95
    random_state: int = 42
    dim3_tuning_mode: str = "shared"
    notes: str = ""

    def __post_init__(self) -> None:
        if 0 not in self.dimensions:
            raise ValueError(
                "PipelineConfig: dimension 0 must be included in 'dimensions' — "
                "it produces clean_df required by all downstream dimensions."
            )
        if self.n_boot < 0:
            raise ValueError(f"n_boot must be >= 0, got {self.n_boot}.")
        if not (0.0 < self.boot_ci < 1.0):
            raise ValueError(f"boot_ci must be in (0, 1), got {self.boot_ci}.")
        if self.dim3_tuning_mode not in ("shared", "independent"):
            raise ValueError(
                f"dim3_tuning_mode must be 'shared' or 'independent', "
                f"got '{self.dim3_tuning_mode}'."
            )

"""
llm_gtd_benchmark
=================
A rigorous benchmark suite for LLM-based tabular data generation models.

Typical workflow
----------------
>>> from llm_gtd_benchmark import (
...     DataSchema, StructuralInterceptor, FidelityEvaluator,
...     LogicEvaluator, MLUtilityEvaluator, TaskType,
...     PrivacyEvaluator, FairnessEvaluator, FairSpec,
...     BenchmarkPipeline, PipelineConfig,
...     ResultBundle, BundleMetadata,
...     DatasetProfiler, SignificanceTester,
... )
>>> schema  = DataSchema(real_df)
>>> result0 = StructuralInterceptor(schema).evaluate(synth_df)
>>> result1 = FidelityEvaluator(schema, real_df).evaluate(result0.clean_df)
>>> result2 = LogicEvaluator(schema, real_df).evaluate(result0.clean_df)
>>> result3 = MLUtilityEvaluator(schema, "label", TaskType.BINARY_CLASS).evaluate(
...     result0.clean_df, test_real_df, train_real_df
... )
>>> result4 = PrivacyEvaluator(schema, train_real_df).evaluate(result0.clean_df)
>>> spec5   = FairSpec(["gender", "race"], "label", TaskType.BINARY_CLASS)
>>> result5 = FairnessEvaluator(schema, spec5, train_real_df).evaluate(
...     result0.clean_df, test_real_df
... )
>>> print(result1.summary)
>>> print(result3.summary)
>>> print(result5.summary)
"""

from llm_gtd_benchmark.core.schema import DataSchema
from llm_gtd_benchmark.core.logic_spec import LogicSpec, discover_fds
from llm_gtd_benchmark.core.exceptions import (
    BenchmarkError,
    GenerationCollapseError,
    SchemaMismatchError,
    InsufficientDataError,
)
from llm_gtd_benchmark.metrics.dimension0 import StructuralInterceptor, Dim0Result
from llm_gtd_benchmark.metrics.dimension1 import FidelityEvaluator, Dim1Result
from llm_gtd_benchmark.metrics.dimension2 import LogicEvaluator, Dim2Result
from llm_gtd_benchmark.metrics.dimension3 import (
    MLUtilityEvaluator,
    LLMUtilityEvaluator,
    Dim3Result,
    TaskType,
)
from llm_gtd_benchmark.metrics.dimension4 import (
    PrivacyEvaluator,
    MemorizationProbe,
    Dim4Result,
    DataCopyingWarning,
)
from llm_gtd_benchmark.metrics.dimension5 import (
    FairnessEvaluator,
    FairSpec,
    Dim5Result,
    GroupCollapseWarning,
)
from llm_gtd_benchmark.visualization.aggregator import ResultAggregator
from llm_gtd_benchmark.core.result_bundle import ResultBundle, BundleMetadata, schema_fingerprint
from llm_gtd_benchmark.pipeline.config import PipelineConfig
from llm_gtd_benchmark.pipeline.runner import BenchmarkPipeline
from llm_gtd_benchmark.analysis.profiler import DatasetProfiler, DatasetProfile, ColumnProfile
from llm_gtd_benchmark.analysis.significance import (
    SignificanceTester,
    SignificanceReport,
    MetricTestResult,
)
from llm_gtd_benchmark.models import (
    BaseTabularModel,
    GReaTModel,
    PAFTModel,
    GraFTModel,
    GraDeModel,
)
from llm_gtd_benchmark.models.paft import discover_fd_order

__all__ = [
    # Core
    "DataSchema",
    "LogicSpec",
    "discover_fds",
    # Exceptions
    "BenchmarkError",
    "GenerationCollapseError",
    "SchemaMismatchError",
    "InsufficientDataError",
    # Dim 0
    "StructuralInterceptor",
    "Dim0Result",
    # Dim 1
    "FidelityEvaluator",
    "Dim1Result",
    # Dim 2
    "LogicEvaluator",
    "Dim2Result",
    # Dim 3
    "MLUtilityEvaluator",
    "LLMUtilityEvaluator",
    "Dim3Result",
    "TaskType",
    # Dim 4
    "PrivacyEvaluator",
    "MemorizationProbe",
    "Dim4Result",
    "DataCopyingWarning",
    # Dim 5
    "FairnessEvaluator",
    "FairSpec",
    "Dim5Result",
    "GroupCollapseWarning",
    # Visualization
    "ResultAggregator",
    # Serialization
    "ResultBundle",
    "BundleMetadata",
    "schema_fingerprint",
    # Pipeline
    "BenchmarkPipeline",
    "PipelineConfig",
    # Analysis
    "DatasetProfiler",
    "DatasetProfile",
    "ColumnProfile",
    "SignificanceTester",
    "SignificanceReport",
    "MetricTestResult",
    # Models
    "BaseTabularModel",
    "GReaTModel",
    "PAFTModel",
    "GraFTModel",
    "GraDeModel",
    "discover_fd_order",
]

__version__ = "0.1.0"

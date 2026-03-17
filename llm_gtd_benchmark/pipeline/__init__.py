"""
llm_gtd_benchmark.pipeline
===========================
Unified experiment runner for multi-dimension benchmark evaluation.

Typical usage
-------------
>>> from llm_gtd_benchmark.pipeline import BenchmarkPipeline, PipelineConfig
>>> config = PipelineConfig(
...     schema=schema,
...     train_real_df=train_df,
...     test_real_df=test_df,
...     model_name="GReaT",
...     dataset_name="adult",
...     target_col="income",
...     task_type=TaskType.BINARY_CLASS,
...     logic_spec=logic_spec,
...     fair_spec=fair_spec,
...     n_boot=1000,
... )
>>> bundle = BenchmarkPipeline(config).run(synth_df)
>>> bundle.save("results/great_adult.json")
"""

from llm_gtd_benchmark.pipeline.config import PipelineConfig
from llm_gtd_benchmark.pipeline.runner import BenchmarkPipeline

__all__ = ["PipelineConfig", "BenchmarkPipeline"]

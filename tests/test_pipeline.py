"""End-to-end BenchmarkPipeline tests (no generative model required)."""

import numpy as np
import pandas as pd
import pytest

from llm_gtd_benchmark import BenchmarkPipeline, DataSchema, PipelineConfig
from llm_gtd_benchmark.metrics.dimension3 import TaskType


class TestBenchmarkPipeline:
    """Run the full pipeline using the real DataFrame as stand-in synth data."""

    def _make_config(self, real_df, test_df):
        schema = DataSchema(real_df)
        return PipelineConfig(
            schema=schema,
            train_real_df=real_df,
            test_real_df=test_df,
            model_name="test_model",
            dataset_name="test_dataset",
            target_col="label",
            task_type=TaskType.BINARY_CLASS,
            n_boot=0,
            random_state=42,
        )

    def test_pipeline_completes(self, small_df, test_df):
        config = self._make_config(small_df, test_df)
        bundle = BenchmarkPipeline(config).run(small_df.copy())
        assert bundle is not None

    def test_bundle_has_dim0_result(self, small_df, test_df):
        config = self._make_config(small_df, test_df)
        bundle = BenchmarkPipeline(config).run(small_df.copy())
        assert bundle.result0 is not None

    def test_bundle_has_dim1_result(self, small_df, test_df):
        config = self._make_config(small_df, test_df)
        bundle = BenchmarkPipeline(config).run(small_df.copy())
        assert bundle.result1 is not None

    def test_bundle_has_dim3_result(self, small_df, test_df):
        config = self._make_config(small_df, test_df)
        bundle = BenchmarkPipeline(config).run(small_df.copy())
        assert bundle.result3 is not None

    def test_bundle_dimensions_computed_non_empty(self, small_df, test_df):
        config = self._make_config(small_df, test_df)
        bundle = BenchmarkPipeline(config).run(small_df.copy())
        assert len(bundle.dimensions_computed) > 0

    def test_no_fatal_errors_on_clean_data(self, small_df, test_df):
        config = self._make_config(small_df, test_df)
        bundle = BenchmarkPipeline(config).run(small_df.copy())
        fatal  = {
            k: v for k, v in bundle.errors.items()
            if not v.startswith("skipped:")
        }
        assert len(fatal) == 0, f"Fatal pipeline errors: {fatal}"

"""Tests for analysis modules: DatasetProfiler and SignificanceTester."""

import numpy as np
import pandas as pd
import pytest

from llm_gtd_benchmark.analysis.profiler import DatasetProfiler, DatasetProfile, ColumnProfile
from llm_gtd_benchmark.analysis.significance import SignificanceTester


class TestDatasetProfiler:
    def test_returns_dataset_profile(self, small_df):
        profile = DatasetProfiler().profile(small_df)
        assert isinstance(profile, DatasetProfile)

    def test_column_count_matches(self, small_df):
        profile = DatasetProfiler().profile(small_df)
        assert profile.n_cols == len(small_df.columns)

    def test_row_count_matches(self, small_df):
        profile = DatasetProfiler().profile(small_df)
        assert profile.n_rows == len(small_df)

    def test_each_column_has_profile(self, small_df):
        profile = DatasetProfiler().profile(small_df)
        for col in small_df.columns:
            assert col in profile.column_profiles

    def test_numeric_column_has_finite_mean(self, small_df):
        profile     = DatasetProfiler().profile(small_df)
        col_profile = profile.column_profiles["age"]
        assert np.isfinite(col_profile.mean)

    def test_categorical_column_has_top_value(self, small_df):
        profile     = DatasetProfiler().profile(small_df)
        col_profile = profile.column_profiles["gender"]
        assert col_profile.top_value in ("Male", "Female")
        assert col_profile.n_unique == 2


class TestSignificanceTester:
    """SignificanceTester compares two lists of ResultBundles.

    We test its public contract with two identical single-run bundles.
    """

    def _make_bundle(self, small_df, test_df, dataset_name="ds"):
        from llm_gtd_benchmark import BenchmarkPipeline, DataSchema, PipelineConfig
        from llm_gtd_benchmark.metrics.dimension3 import TaskType

        schema = DataSchema(small_df)
        config = PipelineConfig(
            schema=schema,
            train_real_df=small_df,
            test_real_df=test_df,
            model_name="m",
            dataset_name=dataset_name,
            target_col="label",
            task_type=TaskType.BINARY_CLASS,
            n_boot=0,
            random_state=42,
        )
        return BenchmarkPipeline(config).run(small_df.copy())

    def test_compare_same_bundles_returns_report(self, small_df, test_df):
        bundle = self._make_bundle(small_df, test_df)
        report = SignificanceTester().compare([bundle], [bundle], model_a="A", model_b="B")
        assert report is not None

    def test_report_has_results_dict(self, small_df, test_df):
        bundle = self._make_bundle(small_df, test_df)
        report = SignificanceTester().compare([bundle], [bundle])
        assert isinstance(report.results, dict)
        assert len(report.results) > 0

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError):
            SignificanceTester(alpha=1.5)

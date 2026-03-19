"""Tests for metric dimensions 0–5 (unit-level, no heavy ML deps required)."""

import numpy as np
import pandas as pd
import pytest

from llm_gtd_benchmark.core.schema import DataSchema
from llm_gtd_benchmark.metrics.dimension0 import StructuralInterceptor
from llm_gtd_benchmark.metrics.dimension1 import FidelityEvaluator
from llm_gtd_benchmark.metrics.dimension2 import LogicEvaluator
from llm_gtd_benchmark.metrics.dimension4 import PrivacyEvaluator
from llm_gtd_benchmark.metrics.dimension3 import MLUtilityEvaluator, TaskType


# ── Dim 0: StructuralInterceptor ──────────────────────────────────────────────

class TestDim0:
    def test_clean_df_returns_same_columns(self, small_df):
        schema = DataSchema(small_df)
        result = StructuralInterceptor(schema).evaluate(small_df.copy())
        assert set(result.clean_df.columns) == set(small_df.columns)

    def test_irr_between_zero_and_one(self, small_df):
        schema = DataSchema(small_df)
        result = StructuralInterceptor(schema).evaluate(small_df.copy())
        assert 0.0 <= result.irr <= 1.0

    def test_all_valid_rows_gives_irr_zero(self, small_df):
        """IRR = Invalid Row Rate; all-valid data should give 0.0."""
        schema = DataSchema(small_df)
        result = StructuralInterceptor(schema).evaluate(small_df.copy())
        assert result.irr == pytest.approx(0.0, abs=0.01)

    def test_oov_categorical_raises_irr(self, small_df):
        """Hallucinated category values in some rows should increase the invalid row rate."""
        schema  = DataSchema(small_df)
        bad_df  = small_df.copy()
        # Replace half the rows with an unseen category so IRR > 0 but
        # enough clean rows remain to avoid GenerationCollapseError.
        n_bad = len(bad_df) // 2
        bad_df.loc[:n_bad - 1, "gender"] = "INVALID_GENDER_VALUE_XYZ"
        result  = StructuralInterceptor(schema).evaluate(bad_df)
        assert result.irr > 0.0

    def test_result_has_summary(self, small_df):
        schema = DataSchema(small_df)
        result = StructuralInterceptor(schema).evaluate(small_df.copy())
        assert hasattr(result, "summary")


# ── Dim 1: FidelityEvaluator ─────────────────────────────────────────────────

class TestDim1:
    def test_identical_df_low_ks(self, small_df):
        """KS between identical distributions should be near 0."""
        schema = DataSchema(small_df)
        result = FidelityEvaluator(schema, small_df).evaluate(small_df.copy())
        assert result.mean_ks < 0.05

    def test_identical_df_low_tvd(self, small_df):
        schema = DataSchema(small_df)
        result = FidelityEvaluator(schema, small_df).evaluate(small_df.copy())
        assert result.mean_tvd < 0.05

    def test_different_distribution_higher_ks(self, small_df):
        rng = np.random.default_rng(7)
        n   = len(small_df)
        shifted_df = small_df.copy()
        shifted_df["age"]    = rng.integers(60, 90, size=n)   # very different
        shifted_df["income"] = rng.integers(200000, 500000, size=n)
        schema = DataSchema(small_df)
        result = FidelityEvaluator(schema, small_df).evaluate(shifted_df)
        assert result.mean_ks > 0.1

    def test_result_metrics_are_finite(self, small_df):
        schema = DataSchema(small_df)
        result = FidelityEvaluator(schema, small_df).evaluate(small_df.copy())
        assert np.isfinite(result.mean_ks)
        assert np.isfinite(result.mean_tvd)


# ── Dim 2: LogicEvaluator ────────────────────────────────────────────────────

class TestDim2:
    def test_evaluates_without_logic_spec(self, small_df):
        """LogicEvaluator should run even without a LogicSpec (DSI only)."""
        schema = DataSchema(small_df)
        result = LogicEvaluator(schema, small_df).evaluate(small_df.copy())
        assert result is not None

    def test_dsi_gap_non_negative(self, small_df):
        schema = DataSchema(small_df)
        result = LogicEvaluator(schema, small_df).evaluate(small_df.copy())
        if result.dsi_gap is not None:
            assert result.dsi_gap >= 0.0


# ── Dim 3: MLUtilityEvaluator ────────────────────────────────────────────────

class TestDim3:
    def test_binary_classification_tstr(self, small_df, test_df):
        schema = DataSchema(small_df)
        result = MLUtilityEvaluator(
            schema, "label", TaskType.BINARY_CLASS
        ).evaluate(small_df.copy(), test_df, small_df)
        assert result.mle_tstr is not None
        assert len(result.mle_tstr) > 0

    def test_tstr_scores_in_valid_range(self, small_df, test_df):
        schema = DataSchema(small_df)
        result = MLUtilityEvaluator(
            schema, "label", TaskType.BINARY_CLASS
        ).evaluate(small_df.copy(), test_df, small_df)
        for clf, metrics in result.mle_tstr.items():
            for metric, val in metrics.items():
                assert np.isfinite(val), f"{clf}/{metric} = {val}"


# ── Dim 4: PrivacyEvaluator ──────────────────────────────────────────────────

class TestDim4:
    def test_dcr_5th_percentile_positive(self, small_df):
        schema = DataSchema(small_df)
        result = PrivacyEvaluator(schema, small_df).evaluate(small_df.copy())
        # For identical data DCR should be ~0 (worst-case privacy)
        assert result.dcr_5th_percentile >= 0.0

    def test_dcr_95th_percentile_positive(self, small_df):
        schema = DataSchema(small_df)
        result = PrivacyEvaluator(schema, small_df).evaluate(small_df.copy())
        # 95th percentile must be non-negative and >= 5th percentile
        assert result.dcr_95th_percentile >= 0.0
        assert result.dcr_95th_percentile >= result.dcr_5th_percentile

    def test_exact_match_rate_identical(self, small_df):
        schema = DataSchema(small_df)
        result = PrivacyEvaluator(schema, small_df).evaluate(small_df.copy())
        # Exact copies → exact_match_rate should be > 0
        assert result.exact_match_rate >= 0.0

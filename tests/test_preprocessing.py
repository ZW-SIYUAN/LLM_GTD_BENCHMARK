"""Tests for llm_gtd_benchmark.utils.preprocessing."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from llm_gtd_benchmark.core.schema import DataSchema
from llm_gtd_benchmark.utils.preprocessing import build_feature_encoder, stratified_subsample


class TestBuildFeatureEncoder:
    def test_returns_column_transformer(self, small_df):
        schema  = DataSchema(small_df)
        encoder = build_feature_encoder(schema, small_df)
        assert isinstance(encoder, ColumnTransformer)

    def test_transform_shape(self, small_df):
        schema  = DataSchema(small_df)
        encoder = build_feature_encoder(schema, small_df)
        arr     = encoder.transform(small_df)
        assert arr.shape[0] == len(small_df)
        assert arr.shape[1] > 0

    def test_transform_no_nans(self, small_df):
        schema  = DataSchema(small_df)
        encoder = build_feature_encoder(schema, small_df)
        arr     = encoder.transform(small_df)
        assert not np.isnan(arr).any()

    def test_oov_category_no_error(self, small_df):
        """OOV categories in synthetic data should silently produce zero rows."""
        schema  = DataSchema(small_df)
        encoder = build_feature_encoder(schema, small_df)
        synth   = small_df.copy()
        synth.loc[0, "gender"] = "NonBinary"  # not in training data
        arr = encoder.transform(synth)
        assert arr.shape[0] == len(synth)
        assert not np.isnan(arr).any()

    def test_empty_schema_raises(self):
        schema = DataSchema(pd.DataFrame())
        with pytest.raises(ValueError):
            build_feature_encoder(schema, pd.DataFrame())


class TestStratifiedSubsample:
    def test_small_df_returned_unchanged(self, small_df):
        out = stratified_subsample(small_df, max_rows=10_000)
        assert len(out) == len(small_df)

    def test_respects_max_rows(self, small_df):
        max_rows = 50
        out = stratified_subsample(small_df, max_rows=max_rows)
        assert len(out) <= max_rows

    def test_stratified_preserves_label_ratio(self, small_df):
        real_ratio    = small_df["label"].mean()
        max_rows      = 80
        out           = stratified_subsample(small_df, max_rows=max_rows, strat_col="label")
        sampled_ratio = out["label"].mean()
        assert abs(sampled_ratio - real_ratio) < 0.15

    def test_missing_strat_col_falls_back(self, small_df):
        out = stratified_subsample(small_df, max_rows=50, strat_col="nonexistent_col")
        assert len(out) <= 50

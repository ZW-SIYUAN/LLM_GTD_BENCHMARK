"""Tests for core modules: DataSchema, ResultBundle."""

import json
import tempfile

import numpy as np
import pandas as pd
import pytest

from llm_gtd_benchmark.core.schema import DataSchema
from llm_gtd_benchmark.core.result_bundle import ResultBundle, BundleMetadata, schema_fingerprint


# ── DataSchema ────────────────────────────────────────────────────────────────

class TestDataSchema:
    def test_infers_continuous_columns(self, small_df):
        schema     = DataSchema(small_df)
        cont_names = [c.name for c in schema.continuous_columns]
        assert "age"    in cont_names
        assert "income" in cont_names

    def test_infers_categorical_columns(self, small_df):
        schema    = DataSchema(small_df)
        cat_names = [c.name for c in schema.categorical_columns]
        assert "gender" in cat_names

    def test_all_columns_accounted(self, small_df):
        schema    = DataSchema(small_df)
        all_names = set(c.name for c in schema.columns)
        assert all_names == set(small_df.columns)

    def test_len_equals_column_count(self, small_df):
        schema = DataSchema(small_df)
        assert len(schema) == len(small_df.columns)

    def test_repr_contains_column_count(self, small_df):
        schema = DataSchema(small_df)
        assert str(len(schema)) in repr(schema)

    def test_column_names_property(self, small_df):
        schema = DataSchema(small_df)
        assert set(schema.column_names) == set(small_df.columns)

    def test_getitem_by_name(self, small_df):
        schema = DataSchema(small_df)
        col    = schema["age"]
        assert col.name     == "age"
        assert col.col_type == "continuous"

    def test_getitem_missing_raises(self, small_df):
        schema = DataSchema(small_df)
        with pytest.raises(KeyError):
            _ = schema["no_such_column"]

    def test_continuous_col_has_bounds(self, small_df):
        schema = DataSchema(small_df)
        col    = schema["age"]
        assert col.min_val is not None
        assert col.max_val is not None
        assert col.min_val <= col.max_val


# ── schema_fingerprint ────────────────────────────────────────────────────────

class TestSchemaFingerprint:
    def test_stable_across_calls(self, small_df):
        schema = DataSchema(small_df)
        assert schema_fingerprint(schema) == schema_fingerprint(schema)

    def test_different_schemas_differ(self, small_df):
        df2 = small_df.drop(columns=["gender"])
        s1  = DataSchema(small_df)
        s2  = DataSchema(df2)
        assert schema_fingerprint(s1) != schema_fingerprint(s2)

    def test_returns_string(self, small_df):
        schema = DataSchema(small_df)
        fp     = schema_fingerprint(schema)
        assert isinstance(fp, str)
        assert len(fp) > 0


# ── ResultBundle serialisation ────────────────────────────────────────────────

class TestResultBundle:
    def _make_bundle(self, small_df):
        schema   = DataSchema(small_df)
        metadata = BundleMetadata(
            model_name="test_model",
            dataset_name="test_dataset",
            schema_fingerprint=schema_fingerprint(schema),
        )
        return ResultBundle(metadata=metadata)

    def test_save_and_load(self, small_df):
        bundle = self._make_bundle(small_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            path   = f"{tmpdir}/bundle.json"
            bundle.save(path)
            loaded = ResultBundle.load(path)
        assert loaded.metadata.model_name   == "test_model"
        assert loaded.metadata.dataset_name == "test_dataset"

    def test_saved_file_is_valid_json(self, small_df):
        bundle = self._make_bundle(small_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/bundle.json"
            bundle.save(path)
            with open(path) as f:
                data = json.load(f)
        assert "metadata" in data

    def test_errors_dict_initially_empty(self, small_df):
        bundle = self._make_bundle(small_df)
        assert isinstance(bundle.errors, dict)
        assert len(bundle.errors) == 0

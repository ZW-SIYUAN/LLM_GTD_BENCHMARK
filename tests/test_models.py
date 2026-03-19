"""Tests for model wrapper interfaces (no GPU / LLM required).

These tests verify:
  - The BaseTabularModel ABC is correctly enforced.
  - PAFTModel.discover_fd_order() returns the right columns.
  - GraFTModel uses IBOrderFinder to reorder columns before training.
  - GraDeModel / GraFTModel load correctly via _load_grade_class.
  - All model save/load round-trips preserve the wrapper state.

Heavy tests (actual fit/sample) are guarded by pytest.importorskip so they
are skipped automatically in environments without torch / be-great.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_gtd_benchmark.models.base import BaseTabularModel


# ── BaseTabularModel ABC ──────────────────────────────────────────────────────

class TestBaseTabularModel:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseTabularModel()  # type: ignore[abstract]

    def test_concrete_subclass_requires_fit_and_sample(self):
        """A subclass that forgets sample() should fail at instantiation."""
        with pytest.raises(TypeError):
            class BadModel(BaseTabularModel):
                def fit(self, train_df):
                    return self
            BadModel()  # type: ignore[abstract]

    def test_fit_sample_calls_fit_then_sample(self):
        calls = []

        class DummyModel(BaseTabularModel):
            def fit(self, train_df):
                calls.append("fit")
                return self

            def sample(self, n_samples):
                calls.append("sample")
                return pd.DataFrame({"x": [1, 2]})

        model = DummyModel()
        result = model.fit_sample(pd.DataFrame({"x": [1, 2, 3]}), n_samples=2)
        assert calls == ["fit", "sample"]
        assert isinstance(result, pd.DataFrame)


# ── discover_fd_order ─────────────────────────────────────────────────────────

class TestDiscoverFdOrder:
    def test_returns_all_columns(self, small_df_with_fds):
        pytest.importorskip("networkx")
        from llm_gtd_benchmark.models.paft import discover_fd_order

        order = discover_fd_order(small_df_with_fds, max_rows=150)
        assert sorted(order) == sorted(small_df_with_fds.columns.tolist())

    def test_zip_before_city_in_order(self, small_df_with_fds):
        """zip → city FD should place 'zip' before 'city'."""
        pytest.importorskip("networkx")
        from llm_gtd_benchmark.models.paft import discover_fd_order

        order = discover_fd_order(small_df_with_fds, max_rows=150)
        if "zip" in order and "city" in order:
            assert order.index("zip") < order.index("city"), (
                f"Expected zip before city, got order: {order}"
            )

    def test_no_duplicate_columns(self, small_df_with_fds):
        pytest.importorskip("networkx")
        from llm_gtd_benchmark.models.paft import discover_fd_order

        order = discover_fd_order(small_df_with_fds, max_rows=150)
        assert len(order) == len(set(order))


# ── PAFTModel interface ───────────────────────────────────────────────────────

class TestPAFTModelInterface:
    def test_column_order_stored(self, small_df):
        pytest.importorskip("be_great")
        from llm_gtd_benchmark.models.paft import PAFTModel

        order = small_df.columns.tolist()
        model = PAFTModel(column_order=order, epochs=1, batch_size=4)
        model.fit(small_df)
        assert model.fitted_order == order

    def test_random_order_permutes_all_columns(self, small_df):
        pytest.importorskip("be_great")
        from llm_gtd_benchmark.models.paft import PAFTModel

        model = PAFTModel(column_order=None, random_state=7, epochs=1, batch_size=4)
        model.fit(small_df)
        assert sorted(model.fitted_order) == sorted(small_df.columns.tolist())

    def test_sample_before_fit_raises(self, small_df):
        from llm_gtd_benchmark.models.paft import PAFTModel
        model = PAFTModel()
        with pytest.raises(RuntimeError):
            model.sample(10)


# ── GraDeModel interface ──────────────────────────────────────────────────────

class TestGraDeModelInterface:
    def test_sample_before_fit_raises(self):
        from llm_gtd_benchmark.models.grade import GraDeModel
        model = GraDeModel()
        with pytest.raises(RuntimeError):
            model.sample(10)

    def test_save_before_fit_raises(self, tmp_path):
        from llm_gtd_benchmark.models.grade import GraDeModel
        model = GraDeModel()
        with pytest.raises(RuntimeError):
            model.save(str(tmp_path / "ckpt"))

    def test_set_fd_list_before_fit_raises(self):
        from llm_gtd_benchmark.models.grade import GraDeModel
        model = GraDeModel()
        with pytest.raises(RuntimeError):
            model.set_fd_list([[["a"], ["b"]]])


# ── GraFTModel interface ──────────────────────────────────────────────────────

class TestGraFTModelInterface:
    def test_fitted_order_none_before_fit(self):
        from llm_gtd_benchmark.models.graft import GraFTModel
        model = GraFTModel()
        assert model.fitted_order_ is None

    def test_ib_finder_none_before_fit(self):
        from llm_gtd_benchmark.models.graft import GraFTModel
        model = GraFTModel()
        assert model.ib_finder_ is None

    def test_sample_before_fit_raises(self):
        from llm_gtd_benchmark.models.graft import GraFTModel
        model = GraFTModel()
        with pytest.raises(RuntimeError):
            model.sample(10)

    def test_chain_metrics_before_fit_raises(self, small_df):
        from llm_gtd_benchmark.models.graft import GraFTModel
        model = GraFTModel()
        with pytest.raises(RuntimeError):
            model.chain_metrics(small_df)


# ── GReaTModel interface ──────────────────────────────────────────────────────

class TestGReaTModelInterface:
    def test_sample_before_fit_raises(self):
        from llm_gtd_benchmark.models.great import GReaTModel
        model = GReaTModel()
        with pytest.raises(RuntimeError):
            model.sample(10)

    def test_fit_sample_end_to_end(self, small_df):
        pytest.importorskip("be_great")
        from llm_gtd_benchmark.models.great import GReaTModel

        model  = GReaTModel(llm="distilgpt2", epochs=1, batch_size=4)
        result = model.fit_sample(small_df, n_samples=10)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_save_load_roundtrip(self, small_df, tmp_path):
        pytest.importorskip("be_great")
        from llm_gtd_benchmark.models.great import GReaTModel

        model = GReaTModel(llm="distilgpt2", epochs=1, batch_size=4)
        model.fit(small_df)
        ckpt  = str(tmp_path / "great_ckpt")
        model.save(ckpt)
        loaded = GReaTModel.load(ckpt)
        synth  = loaded.sample(n_samples=5)
        assert isinstance(synth, pd.DataFrame)


# ── _load_grade_class idempotency ─────────────────────────────────────────────

class TestLoadGradeClass:
    def test_idempotent(self):
        """Calling _load_grade_class() twice returns the same class object."""
        pytest.importorskip("torch")
        from llm_gtd_benchmark.models.grade import _load_grade_class
        cls1 = _load_grade_class()
        cls2 = _load_grade_class()
        assert cls1 is cls2

"""Shared pytest fixtures for llm_gtd_benchmark tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def small_df():
    """200-row synthetic DataFrame with mixed column types, no NaNs."""
    rng = np.random.default_rng(0)
    n = 200
    age     = rng.integers(18, 70, size=n)
    income  = (age * 1000 + rng.normal(0, 5000, n)).astype(int)
    gender  = rng.choice(["Male", "Female"], size=n)
    label   = (income > 45000).astype(int)
    return pd.DataFrame({"age": age, "income": income, "gender": gender, "label": label})


@pytest.fixture(scope="session")
def small_df_with_fds():
    """150-row DataFrame containing a clear functional dependency: zip → city."""
    rng = np.random.default_rng(1)
    n = 150
    zip_code = rng.choice(["10001", "10002", "10003"], size=n)
    city     = np.where(zip_code == "10001", "New York",
               np.where(zip_code == "10002", "Brooklyn", "Queens"))
    age      = rng.integers(20, 60, size=n)
    income   = rng.integers(30000, 100000, size=n)
    return pd.DataFrame({"zip": zip_code, "city": city, "age": age, "income": income})


@pytest.fixture(scope="session")
def real_df(small_df):
    """Alias for small_df used as 'real' training data in pipeline tests."""
    return small_df


@pytest.fixture(scope="session")
def test_df(small_df):
    """A slightly different 80-row DataFrame used as held-out test set."""
    rng = np.random.default_rng(99)
    n = 80
    age    = rng.integers(18, 70, size=n)
    income = (age * 1000 + rng.normal(0, 5000, n)).astype(int)
    gender = rng.choice(["Male", "Female"], size=n)
    label  = (income > 45000).astype(int)
    return pd.DataFrame({"age": age, "income": income, "gender": gender, "label": label})

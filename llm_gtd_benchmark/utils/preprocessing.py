"""
Shared preprocessing utilities.

build_feature_encoder
    Constructs and fits a sklearn ColumnTransformer on the real DataFrame.
    Continuous columns → StandardScaler.
    Categorical columns → OneHotEncoder (handle_unknown='ignore').
    The fitted transformer is the single encoding contract shared between
    all Dimension 1 sub-metrics.

stratified_subsample
    OOM-safe subsampling for datasets larger than a configurable row limit.
    Uses sklearn train_test_split with stratification when a target column is
    available; falls back to uniform random sampling on failure.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from llm_gtd_benchmark.core.schema import DataSchema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature encoder
# ---------------------------------------------------------------------------


def build_feature_encoder(
    schema: DataSchema,
    real_df: pd.DataFrame,
) -> ColumnTransformer:
    """Build and fit a ColumnTransformer from a DataSchema.

    Encoding rules
    --------------
    - Continuous  →  StandardScaler  (zero mean, unit variance)
    - Categorical →  OneHotEncoder   (sparse=False, unknown categories
                                      silently produce all-zero rows)

    The transformer is fitted on *real_df* only.  When applied to synthetic
    data via `.transform()`, any OOV category values produce all-zero
    one-hot rows — a neutral encoding that does not inflate distances.

    Parameters
    ----------
    schema:     DataSchema built from real_df.
    real_df:    The reference training DataFrame.

    Returns
    -------
    A fitted sklearn ColumnTransformer.
    """
    cont_names = [c.name for c in schema.continuous_columns]
    cat_names = [c.name for c in schema.categorical_columns]

    transformers = []
    if cont_names:
        transformers.append(("continuous", StandardScaler(), cont_names))
    if cat_names:
        ohe = _make_ohe()
        transformers.append(("categorical", ohe, cat_names))

    if not transformers:
        raise ValueError("Schema has no columns — cannot build a feature encoder.")

    encoder = ColumnTransformer(transformers=transformers, remainder="drop")
    encoder.fit(real_df)
    logger.debug(
        "Feature encoder fitted: %d continuous, %d categorical columns.",
        len(cont_names),
        len(cat_names),
    )
    return encoder


def _make_ohe() -> OneHotEncoder:
    """Instantiate OneHotEncoder, handling the sparse_output rename in sklearn 1.2."""
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------


def stratified_subsample(
    df: pd.DataFrame,
    max_rows: int,
    strat_col: Optional[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a subset of *df* with at most *max_rows* rows.

    If the DataFrame already fits within the limit, it is returned as-is
    (no copy).

    Sampling strategy (in order of preference)
    -------------------------------------------
    1. Stratified sampling via sklearn's ``train_test_split`` when *strat_col*
       is provided and its class sizes allow it.
    2. Uniform random sampling as a fallback for any failure mode (class too
       small, strat_col missing, etc.).

    Parameters
    ----------
    df:             Source DataFrame.
    max_rows:       Hard upper bound on returned rows.
    strat_col:      Column name to stratify on (e.g. target label).
    random_state:   Reproducibility seed.

    Returns
    -------
    A reset-index DataFrame of at most *max_rows* rows.
    """
    if len(df) <= max_rows:
        return df

    sample_fraction = max_rows / len(df)

    if strat_col is not None and strat_col in df.columns:
        try:
            _, sampled = train_test_split(
                df,
                test_size=sample_fraction,
                stratify=df[strat_col],
                random_state=random_state,
            )
            logger.debug(
                "Stratified subsample: %d → %d rows (strat_col='%s').",
                len(df),
                len(sampled),
                strat_col,
            )
            return sampled.reset_index(drop=True)
        except ValueError as exc:
            warnings.warn(
                f"Stratified sampling on '{strat_col}' failed ({exc}); "
                "falling back to uniform random sampling.",
                RuntimeWarning,
                stacklevel=2,
            )

    sampled = df.sample(n=max_rows, random_state=random_state)
    logger.debug("Uniform subsample: %d → %d rows.", len(df), len(sampled))
    return sampled.reset_index(drop=True)

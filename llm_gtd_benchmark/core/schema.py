"""
DataSchema — the single source of truth for column metadata.

Built once from the real training DataFrame; consumed by every downstream
evaluator (Dimension 0, Dimension 1, …) to ensure all modules operate on
identical type and domain information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ColType = Literal["continuous", "categorical"]

# Columns with at most this many unique values are auto-detected as categorical
# when no explicit override is provided.
_AUTO_CATEGORICAL_THRESHOLD: int = 20


@dataclass(frozen=True)
class ColumnSchema:
    """Immutable metadata for a single column.

    Attributes
    ----------
    name:       Column name.
    col_type:   "continuous" or "categorical".
    dtype:      Original numpy dtype from the real DataFrame.
    min_val:    Minimum value (continuous only).
    max_val:    Maximum value (continuous only).
    categories: Frozenset of all observed values (categorical only).
    """

    name: str
    col_type: ColType
    dtype: np.dtype
    # continuous
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    # categorical
    categories: Optional[FrozenSet] = None

    def __post_init__(self) -> None:
        if self.col_type == "continuous" and (
            self.min_val is None or self.max_val is None
        ):
            raise ValueError(
                f"ColumnSchema '{self.name}': continuous column must supply min_val and max_val."
            )
        if self.col_type == "categorical" and self.categories is None:
            raise ValueError(
                f"ColumnSchema '{self.name}': categorical column must supply categories."
            )


class DataSchema:
    """Immutable schema derived from a real training DataFrame.

    The schema captures, for every column:
    - Whether it is continuous or categorical (auto-detected or user-specified).
    - The original dtype (for strict downcasting in Dimension 0).
    - Domain bounds [min, max] for continuous columns.
    - The closed vocabulary (frozenset of unique values) for categorical columns.

    Parameters
    ----------
    real_df:
        The real training DataFrame used as the reference distribution.
    categorical_columns:
        Optional explicit list of column names to treat as categorical,
        overriding auto-detection.
    continuous_columns:
        Optional explicit list of column names to treat as continuous,
        overriding auto-detection.
    categorical_threshold:
        Auto-detection heuristic: numeric columns with at most this many
        unique values are treated as categorical (default 20).

    Examples
    --------
    >>> schema = DataSchema(real_df)
    >>> schema = DataSchema(real_df, categorical_columns=["zip_code"])
    >>> print(schema)
    DataSchema(10 columns: 6 continuous, 4 categorical)
    """

    def __init__(
        self,
        real_df: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        continuous_columns: Optional[List[str]] = None,
        categorical_threshold: int = _AUTO_CATEGORICAL_THRESHOLD,
    ) -> None:
        explicit_cat: set = set(categorical_columns or [])
        explicit_cont: set = set(continuous_columns or [])

        overlap = explicit_cat & explicit_cont
        if overlap:
            raise ValueError(
                f"Columns declared as both categorical and continuous: {sorted(overlap)}"
            )

        unknown = (explicit_cat | explicit_cont) - set(real_df.columns)
        if unknown:
            raise ValueError(
                f"Declared columns not found in real_df: {sorted(unknown)}"
            )

        self._threshold = categorical_threshold
        self._col_order: List[str] = list(real_df.columns)
        self._columns: Dict[str, ColumnSchema] = {}

        for col in real_df.columns:
            series = real_df[col]
            col_type = self._resolve_col_type(series, col, explicit_cat, explicit_cont)
            self._columns[col] = self._build_col_schema(col, col_type, series)

        n_cont = len(self.continuous_columns)
        n_cat = len(self.categorical_columns)
        logger.debug("DataSchema built: %d continuous, %d categorical columns.", n_cont, n_cat)

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def columns(self) -> List[ColumnSchema]:
        """All column schemas in original DataFrame column order."""
        return [self._columns[c] for c in self._col_order]

    @property
    def continuous_columns(self) -> List[ColumnSchema]:
        """Continuous-only column schemas in original order."""
        return [c for c in self.columns if c.col_type == "continuous"]

    @property
    def categorical_columns(self) -> List[ColumnSchema]:
        """Categorical-only column schemas in original order."""
        return [c for c in self.columns if c.col_type == "categorical"]

    @property
    def column_names(self) -> List[str]:
        return list(self._col_order)

    def __getitem__(self, col_name: str) -> ColumnSchema:
        try:
            return self._columns[col_name]
        except KeyError:
            raise KeyError(f"Column '{col_name}' not found in schema.") from None

    def __len__(self) -> int:
        return len(self._columns)

    def __repr__(self) -> str:
        n_cont = len(self.continuous_columns)
        n_cat = len(self.categorical_columns)
        return f"DataSchema({len(self)} columns: {n_cont} continuous, {n_cat} categorical)"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_col_type(
        self,
        series: pd.Series,
        col: str,
        explicit_cat: set,
        explicit_cont: set,
    ) -> ColType:
        if col in explicit_cat:
            return "categorical"
        if col in explicit_cont:
            return "continuous"
        return self._auto_detect(series, col)

    def _auto_detect(self, series: pd.Series, col: str) -> ColType:
        """Heuristic: object/bool/category dtypes → categorical.
        Numeric with few unique values → categorical.
        Everything else → continuous.
        """
        if series.dtype == object or series.dtype.name == "category" or series.dtype == bool:
            return "categorical"
        n_unique = series.nunique(dropna=True)
        if n_unique <= self._threshold:
            logger.debug(
                "Column '%s': auto-detected as categorical (%d unique values ≤ threshold %d).",
                col,
                n_unique,
                self._threshold,
            )
            return "categorical"
        return "continuous"

    @staticmethod
    def _build_col_schema(col: str, col_type: ColType, series: pd.Series) -> ColumnSchema:
        if col_type == "continuous":
            non_null = series.dropna()
            if len(non_null) == 0:
                raise ValueError(
                    f"Continuous column '{col}' has no non-null values — cannot derive bounds."
                )
            return ColumnSchema(
                name=col,
                col_type="continuous",
                dtype=series.dtype,
                min_val=float(non_null.min()),
                max_val=float(non_null.max()),
            )
        else:
            return ColumnSchema(
                name=col,
                col_type="categorical",
                dtype=series.dtype,
                categories=frozenset(series.dropna().unique()),
            )

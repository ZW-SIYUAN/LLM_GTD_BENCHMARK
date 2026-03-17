"""
Dimension 0 — Structural Validity & Domain Support Interceptor.

Theoretical framing
-------------------
Before any distributional metric can be computed, the generated data must lie
within the *probability support* of the real distribution.  This module acts
as a mandatory pre-filter that projects LLM output back onto the real data's
domain manifold by detecting and discarding three classes of structural defects:

    1. Type-coercion NaN   — values that cannot be parsed as the declared dtype.
    2. OOV hallucination   — categorical values outside the real vocabulary.
    3. Out-of-bounds       — continuous values outside [min, max] of real data.

The fraction of discarded rows is reported as the **Invalid Row Rate (IRR)**,
which is itself a first-order quality signal: a high IRR indicates the model
does not understand the data's structural constraints.

Engineering guarantees
----------------------
- All surviving columns are strictly downcast to the real data's dtype,
  preventing silent type drift in downstream metrics.
- A ``GenerationCollapseError`` is raised (not silently ignored) if fewer
  than ``min_clean_rows`` rows survive, signalling catastrophic failure.
- Defect detection operates row-wise (a single defective value taints the
  entire row), consistent with the semantic interpretation that a row with
  any invalid field is unusable as a synthetic observation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from llm_gtd_benchmark.core.exceptions import GenerationCollapseError, SchemaMismatchError
from llm_gtd_benchmark.core.schema import ColumnSchema, DataSchema

logger = logging.getLogger(__name__)

# Defect-type string constants (also used as dict keys in Dim0Result)
_D_TYPE = "type_coercion_nan"
_D_OOV = "oov_hallucination"
_D_OOB = "out_of_bounds"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Dim0Result:
    """Output of :class:`StructuralInterceptor`.

    Attributes
    ----------
    irr:
        Invalid Row Rate — fraction of generated rows that contained at least
        one structural defect.  Range [0, 1].  Lower is better.
    n_total:
        Total number of rows in the raw synthetic DataFrame.
    n_invalid:
        Number of rows flagged as defective and removed.
    n_clean:
        Number of rows that passed all structural checks.
    defect_counts:
        Breakdown of defective rows by defect category.  A single row may
        exhibit multiple defect types, but it is counted only once in *n_invalid*.
        Keys: ``"type_coercion_nan"``, ``"oov_hallucination"``, ``"out_of_bounds"``.
    clean_df:
        Validated, downcast DataFrame ready for Dimension 1 evaluation.
    """

    irr: float
    n_total: int
    n_invalid: int
    n_clean: int
    defect_counts: Dict[str, int] = field(default_factory=dict)
    clean_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def summary(self) -> str:
        lines = [
            "── Dimension 0: Structural Validity ─────────────────────",
            f"  Total generated rows : {self.n_total:>8,}",
            f"  Invalid rows         : {self.n_invalid:>8,}  ({self.irr:.2%})",
            f"  Clean rows           : {self.n_clean:>8,}",
            "  Defect breakdown:",
            f"    type_coercion_nan  : {self.defect_counts.get(_D_TYPE, 0):>8,}",
            f"    oov_hallucination  : {self.defect_counts.get(_D_OOV, 0):>8,}",
            f"    out_of_bounds      : {self.defect_counts.get(_D_OOB, 0):>8,}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


class StructuralInterceptor:
    """Dimension 0 structural validity gate.

    Reads a raw synthetic DataFrame, validates it against the real data schema,
    and returns a clean DataFrame along with the Invalid Row Rate (IRR).

    Parameters
    ----------
    schema:
        :class:`~llm_gtd_benchmark.core.schema.DataSchema` built from the
        real training data.  Acts as the sole source of truth for column
        types, categories, and numeric bounds.
    min_clean_rows:
        Minimum number of structurally valid rows required before the result
        is considered usable.  If fewer rows survive, a
        :class:`~llm_gtd_benchmark.core.exceptions.GenerationCollapseError`
        is raised.  Default: 100.

    Examples
    --------
    >>> schema     = DataSchema(real_df)
    >>> interceptor = StructuralInterceptor(schema)
    >>> result     = interceptor.evaluate(synth_df)
    >>> print(result.summary)
    >>> clean_df   = result.clean_df          # pass to FidelityEvaluator
    """

    def __init__(self, schema: DataSchema, min_clean_rows: int = 100) -> None:
        self.schema = schema
        self.min_clean_rows = min_clean_rows

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, synth_df: pd.DataFrame) -> Dim0Result:
        """Run structural validation on a raw synthetic DataFrame.

        Parameters
        ----------
        synth_df:
            Raw output from the LLM-based generative model.  May contain any
            combination of type errors, hallucinated categories, and out-of-
            range numeric values.

        Returns
        -------
        :class:`Dim0Result`

        Raises
        ------
        SchemaMismatchError
            If *synth_df*'s column set does not exactly match the schema.
        GenerationCollapseError
            If fewer than ``min_clean_rows`` rows survive filtering.
        """
        self._assert_column_match(synth_df)

        n_total = len(synth_df)

        # invalid_mask accumulates row-level defect flags across all columns.
        invalid_mask = pd.Series(False, index=synth_df.index)
        defect_counts: Dict[str, int] = {_D_TYPE: 0, _D_OOV: 0, _D_OOB: 0}

        # Work on a copy so the original caller's DataFrame is never mutated.
        working = synth_df.copy()

        for col_schema in self.schema.columns:
            col = col_schema.name

            if col_schema.col_type == "continuous":
                type_mask, oob_mask = self._check_continuous(working[col], col_schema)

                # Count only *newly flagged* rows to avoid double-counting
                defect_counts[_D_TYPE] += int((type_mask & ~invalid_mask).sum())
                defect_counts[_D_OOB] += int((oob_mask & ~invalid_mask & ~type_mask).sum())

                invalid_mask |= type_mask | oob_mask

                # Replace the entire column with a properly-typed numeric Series.
                # Rows flagged as invalid will have NaN here, but they are
                # discarded when clean_df is built — so NaN is harmless.
                # This approach avoids pandas FutureWarnings from mixed-dtype
                # partial assignments and guarantees correct dtypes in clean_df.
                coerced_col = pd.to_numeric(working[col], errors="coerce")
                try:
                    working[col] = coerced_col.astype(col_schema.dtype)
                except (ValueError, TypeError):
                    working[col] = coerced_col  # fall back to float64

            else:  # categorical
                oov_mask = self._check_categorical(working[col], col_schema)

                defect_counts[_D_OOV] += int((oov_mask & ~invalid_mask).sum())
                invalid_mask |= oov_mask

                # Attempt to downcast the whole column to the real dtype.
                # For string-encoded integer/float categoricals this will succeed;
                # for pure-string categoricals dtype is already object — no-op.
                try:
                    working[col] = working[col].astype(col_schema.dtype)
                except (ValueError, TypeError):
                    pass  # leave as object; invalid rows will be dropped anyway

        n_invalid = int(invalid_mask.sum())
        n_clean = n_total - n_invalid
        irr = n_invalid / n_total if n_total > 0 else 1.0

        clean_df = working.loc[~invalid_mask].reset_index(drop=True)

        if n_clean < self.min_clean_rows:
            raise GenerationCollapseError(n_clean=n_clean, threshold=self.min_clean_rows)

        logger.info(
            "Dim0 complete: %d/%d rows invalid (IRR=%.2f%%), %d clean rows.",
            n_invalid,
            n_total,
            irr * 100,
            n_clean,
        )

        return Dim0Result(
            irr=irr,
            n_total=n_total,
            n_invalid=n_invalid,
            n_clean=n_clean,
            defect_counts=defect_counts,
            clean_df=clean_df,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _assert_column_match(self, synth_df: pd.DataFrame) -> None:
        schema_cols = {c.name for c in self.schema.columns}
        synth_cols = set(synth_df.columns)
        missing = schema_cols - synth_cols
        extra = synth_cols - schema_cols
        if missing or extra:
            raise SchemaMismatchError(missing=missing, extra=extra)

    @staticmethod
    def _check_continuous(series: pd.Series, col_schema: ColumnSchema):
        """Return (type_invalid_mask, out_of_bounds_mask).

        type_invalid_mask:  rows where the value cannot be parsed as a number.
        out_of_bounds_mask: rows where the parsed value lies outside [min, max].

        Uses numpy arrays for the OOB comparison to avoid pandas FutureWarnings
        caused by assigning a sub-indexed boolean Series to a full boolean Series.
        """
        coerced = pd.to_numeric(series, errors="coerce")

        # Rows that were not already NaN but became NaN after numeric coercion
        type_mask = coerced.isna() & series.notna()

        # Out-of-bounds check on type-valid, non-null rows — via numpy to avoid
        # pandas FutureWarning on boolean Series slice assignment.
        valid_arr = (~type_mask).values & coerced.notna().values
        coerced_vals = coerced.to_numpy(dtype=float, na_value=np.nan)
        oob_arr = np.zeros(len(series), dtype=bool)
        oob_arr[valid_arr] = (
            (coerced_vals[valid_arr] < col_schema.min_val)
            | (coerced_vals[valid_arr] > col_schema.max_val)
        )
        oob_mask = pd.Series(oob_arr, index=series.index)

        return type_mask, oob_mask

    @staticmethod
    def _check_categorical(series: pd.Series, col_schema: ColumnSchema) -> pd.Series:
        """Return a boolean mask: True where the value is out-of-vocabulary.

        Comparison strategy:
        1. Exact string match against the training vocabulary.
        2. For integer-valued categories (e.g., ``{0, 1, ..., 17}``), also
           accept the float representation (``"4.0"`` → normalised to ``"4"``).
           This handles LLM outputs and CSV round-trips that store integer
           columns as floats.

        Case-insensitive matching is NOT applied — label capitalisation
        (``"Yes"`` vs ``"yes"``) is intentionally treated as OOV so that
        inconsistent casing from a generator is flagged as a defect.

        Empty-vocabulary special case:
        When the real training column contained only NaN values, the schema
        stores ``categories = frozenset()``.  In that situation the only valid
        synthetic value is also NaN (preserving the "always missing" real-data
        pattern); any non-NaN value produced by the generator is treated as an
        OOV hallucination.
        """
        # Guard: all-NaN training column → empty vocabulary.
        # NaN in synthetic is valid; any non-NaN is an OOV hallucination.
        if not col_schema.categories:
            return series.notna()

        real_vocab = {str(v) for v in col_schema.categories}

        # Build a normalised vocab that also accepts float representations of
        # integer categories, e.g. "4.0" for category 4.
        expanded_vocab = set(real_vocab)
        for v in col_schema.categories:
            s = str(v)
            # If the canonical string looks like a plain integer, also accept
            # the float form ("4" → also accept "4.0").
            if s.lstrip("-").isdigit():
                expanded_vocab.add(s + ".0")
            # Conversely, if the value is stored as a float integer ("4.0"),
            # accept the int form ("4").  Handles categories inferred from
            # float-typed training columns.
            try:
                f = float(s)
                if f == int(f):
                    expanded_vocab.add(str(int(f)))
            except (ValueError, OverflowError):
                pass

        return ~series.astype(str).isin(expanded_vocab)

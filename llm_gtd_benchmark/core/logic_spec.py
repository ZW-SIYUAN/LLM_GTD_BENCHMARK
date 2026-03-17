"""
LogicSpec — dataset-level semantic metadata for Dimension 2 evaluation.

Separation of concerns
-----------------------
DataSchema  (core/schema.py)   captures STATISTICAL metadata: column types,
                                dtypes, numeric bounds, categorical vocabularies.
LogicSpec   (this file)        captures SEMANTIC/LOGICAL metadata: functional
                                dependencies, hierarchical ontologies, arithmetic
                                identities.  All fields are optional; omitting a
                                field means the corresponding metric is silently
                                skipped (returns NaN) rather than raising an error.

Design philosophy
-----------------
The "Automated Logic Probe & Registry" pattern: instead of hard-coding which
datasets support which metrics, the framework routes computation through the
LogicSpec at runtime.  This allows the same evaluator code to handle any
dataset — known benchmark sets with pre-filled specs, or novel datasets with
user-supplied specs, or datasets with no specs (universal DSI only).

Extending to new constraint types
-----------------------------------
Add a new field to LogicSpec (e.g. ``uniqueness_constraints: List[str]``)
and a corresponding branch in LogicEvaluator._route().  No other files need
to change.

Auto-discovery helper
----------------------
``discover_fds(real_df, schema)`` provides a lightweight scan that surfaces
candidate functional dependencies worth registering in ``LogicSpec.known_fds``.
It does NOT modify the spec — discovery is advisory, registration is explicit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import pandas as pd

from llm_gtd_benchmark.core.schema import DataSchema

logger = logging.getLogger(__name__)

# Type aliases for readability
FDPair = Tuple[str, str]            # (determinant_col, dependent_col)  A → B
HierarchyChain = List[str]          # coarse-to-fine: ["Country","State","City"]
Operator = Literal["+", "-", "*", "/"]
# MathEquation: (col_A, op, col_B, col_result)  →  col_A op col_B ≈ col_result
MathEquation = Tuple[str, Operator, str, str]


@dataclass
class LogicSpec:
    """Semantic constraint registry for a single dataset.

    All fields default to empty lists, which causes the corresponding metric
    to be skipped with a NaN result rather than raising an error.

    Parameters
    ----------
    name:
        Human-readable dataset identifier (used in log messages).
    known_fds:
        Functional dependencies expected to hold strictly in real data.
        Each entry is a ``(determinant, dependent)`` pair meaning "knowing
        the value of *determinant* uniquely determines *dependent*".
        Example: ``[("zip_code", "city"), ("zip_code", "state")]``
    hierarchies:
        Ordered lists of columns forming a coarse-to-fine ontology tree.
        The evaluator checks that (ancestor, descendant) co-occurrences in
        synthetic data are a subset of those observed in real data.
        Example: ``[["country", "state", "city"]]``
    math_equations:
        Deterministic arithmetic identities of the form
        ``col_A  op  col_B  =  col_result``.
        Example: ``[("unit_price", "*", "quantity", "total_price")]``
    fd_violation_tolerance:
        Fraction of FD violations tolerated in *real_df* before the FD is
        considered "soft" (not strictly enforced) and a warning is logged.
        Default 0.0 = strict (any real violation triggers a warning).

    Examples
    --------
    >>> spec = LogicSpec(
    ...     name="adult_income",
    ...     known_fds=[("education_num", "education")],
    ... )
    >>> spec = LogicSpec(name="california_housing")   # no logic constraints → DSI only
    """

    name: str = "unnamed_dataset"
    known_fds: List[FDPair] = field(default_factory=list)
    hierarchies: List[HierarchyChain] = field(default_factory=list)
    math_equations: List[MathEquation] = field(default_factory=list)
    fd_violation_tolerance: float = 0.0

    # ── Applicability checks ──────────────────────────────────────────────────

    def is_applicable(self, metric: str) -> bool:
        """Return True if *metric* can be computed for this spec.

        Logs a structured warning when a metric is skipped, so callers do not
        need to duplicate the log message.

        Parameters
        ----------
        metric: ``"ICVR"``, ``"HCS"``, or ``"MDI"``.
        """
        if metric == "ICVR":
            if not self.known_fds:
                logger.warning(
                    "[SKIP] ICVR: dataset '%s' has no registered functional "
                    "dependencies (known_fds=[]).  Register at least one FD pair "
                    "in LogicSpec to enable this metric.",
                    self.name,
                )
                return False

        elif metric == "HCS":
            if not self.hierarchies:
                logger.warning(
                    "[SKIP] HCS: dataset '%s' has no registered hierarchical "
                    "chains (hierarchies=[]).  Register at least one chain "
                    "in LogicSpec to enable this metric.",
                    self.name,
                )
                return False

        elif metric == "MDI":
            if not self.math_equations:
                logger.warning(
                    "[SKIP] MDI: dataset '%s' has no registered arithmetic "
                    "equations (math_equations=[]).  Register at least one "
                    "equation tuple in LogicSpec to enable this metric.",
                    self.name,
                )
                return False

        else:
            raise ValueError(f"Unknown metric '{metric}'. Expected 'ICVR', 'HCS', or 'MDI'.")

        return True

    def __repr__(self) -> str:
        return (
            f"LogicSpec(name='{self.name}', "
            f"fds={len(self.known_fds)}, "
            f"hierarchies={len(self.hierarchies)}, "
            f"equations={len(self.math_equations)})"
        )


# ---------------------------------------------------------------------------
# Auto-discovery utility
# ---------------------------------------------------------------------------


def discover_fds(
    real_df: pd.DataFrame,
    schema: DataSchema,
    max_unique_determinant: int = 1000,
    min_rows: int = 30,
) -> List[FDPair]:
    """Lightweight scan for candidate functional dependencies in *real_df*.

    Uses the heuristic: if grouping by column A yields exactly 1 unique value
    of column B in every group, then A → B is a functional dependency.

    This is O(d²) in the number of columns but uses pandas groupby which is
    highly optimised.  For datasets with d > 50 columns, consider passing an
    explicit column subset.

    Parameters
    ----------
    real_df:
        The real training DataFrame.
    schema:
        DataSchema for the dataset (used to limit candidates to avoid
        checking high-cardinality continuous → continuous pairs).
    max_unique_determinant:
        Skip columns with more unique values than this as determinants
        (high-cardinality numerics rarely form useful FDs).
    min_rows:
        Minimum number of rows required to trust the FD scan.

    Returns
    -------
    List of ``(determinant, dependent)`` pairs where A → B holds strictly.
    Returns an empty list if the dataset is too small.

    Notes
    -----
    This function is **advisory only**.  It surfaces candidates; a human (or
    domain expert) should verify and register them in LogicSpec.known_fds.
    False positives are possible when a column has very few unique values.
    """
    if len(real_df) < min_rows:
        logger.warning(
            "discover_fds: dataset too small (%d rows < min_rows=%d); returning [].",
            len(real_df),
            min_rows,
        )
        return []

    candidates: List[FDPair] = []
    cols = list(real_df.columns)

    for det in cols:
        n_unique_det = real_df[det].nunique(dropna=True)
        if n_unique_det > max_unique_determinant or n_unique_det < 2:
            continue

        for dep in cols:
            if det == dep:
                continue

            # Count unique dep values within each det group
            max_dep_per_group = real_df.groupby(det, observed=True)[dep].nunique().max()
            if max_dep_per_group == 1:
                candidates.append((det, dep))
                logger.debug("FD candidate discovered: '%s' → '%s'", det, dep)

    if candidates:
        logger.info(
            "discover_fds: found %d candidate FD(s) in '%s'. "
            "Review and register desired pairs in LogicSpec.known_fds.",
            len(candidates),
            real_df.columns.tolist(),
        )
    else:
        logger.info("discover_fds: no strict FD candidates found.")

    return candidates

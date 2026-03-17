"""
Custom exception hierarchy for llm_gtd_benchmark.

All public exceptions inherit from BenchmarkError so callers can catch the
entire family with a single except clause when needed.
"""

from __future__ import annotations


class BenchmarkError(Exception):
    """Base class for all llm-gtd-benchmark exceptions."""


class GenerationCollapseError(BenchmarkError):
    """Raised when structural filtering leaves fewer than the required minimum rows.

    This indicates a catastrophic generation failure — the model produced almost
    entirely invalid output (hallucinated categories, out-of-bounds values, etc.).

    Attributes
    ----------
    n_clean:    Number of rows that survived filtering.
    threshold:  Minimum required clean rows (configurable in StructuralInterceptor).
    """

    def __init__(self, n_clean: int, threshold: int) -> None:
        super().__init__(
            f"Generation collapsed: only {n_clean} valid row(s) remain after structural "
            f"filtering (minimum required: {threshold}). "
            "The model may have produced catastrophically invalid output."
        )
        self.n_clean = n_clean
        self.threshold = threshold


class SchemaMismatchError(BenchmarkError):
    """Raised when synthetic data columns do not match the real-data schema.

    Attributes
    ----------
    missing:  Columns present in the schema but absent from synth_df.
    extra:    Columns present in synth_df but absent from the schema.
    """

    def __init__(self, missing: set, extra: set) -> None:
        parts: list[str] = []
        if missing:
            parts.append(f"missing columns: {sorted(missing)}")
        if extra:
            parts.append(f"extra columns: {sorted(extra)}")
        super().__init__("Column mismatch — " + "; ".join(parts))
        self.missing = missing
        self.extra = extra


class InsufficientDataError(BenchmarkError):
    """Raised when a dataset is too small to compute a specific metric reliably.

    Attributes
    ----------
    metric:     Name of the metric that could not be computed.
    n_samples:  Actual number of samples available.
    minimum:    Minimum samples required for the metric.
    """

    def __init__(self, metric: str, n_samples: int, minimum: int) -> None:
        super().__init__(
            f"Metric '{metric}' requires at least {minimum} samples; got {n_samples}."
        )
        self.metric = metric
        self.n_samples = n_samples
        self.minimum = minimum

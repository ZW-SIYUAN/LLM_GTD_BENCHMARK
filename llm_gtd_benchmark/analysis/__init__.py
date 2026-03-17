"""
llm_gtd_benchmark.analysis
==========================
Dataset profiling and statistical significance testing utilities.

Typical usage
-------------
Dataset profiling::

    >>> from llm_gtd_benchmark.analysis import DatasetProfiler
    >>> profiler = DatasetProfiler()
    >>> profile = profiler.profile(df, dataset_name="adult")
    >>> print(profile.summary)

Multi-run significance testing::

    >>> from llm_gtd_benchmark.analysis import SignificanceTester
    >>> tester = SignificanceTester(alpha=0.05)
    >>> report = tester.compare(bundles_a, bundles_b, model_a="GReaT", model_b="REaLTabFormer")
    >>> print(report.summary)
"""

from llm_gtd_benchmark.analysis.profiler import (
    ColumnProfile,
    DatasetProfile,
    DatasetProfiler,
)
from llm_gtd_benchmark.analysis.significance import (
    MetricTestResult,
    SignificanceReport,
    SignificanceTester,
)

__all__ = [
    # Profiler
    "DatasetProfiler",
    "DatasetProfile",
    "ColumnProfile",
    # Significance
    "SignificanceTester",
    "SignificanceReport",
    "MetricTestResult",
]

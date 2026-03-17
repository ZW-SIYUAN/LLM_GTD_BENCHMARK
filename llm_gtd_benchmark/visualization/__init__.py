"""
llm_gtd_benchmark.visualization
================================
Publication-quality result aggregation and visualisation for multi-model
benchmark comparisons.

Typical usage
-------------
>>> from llm_gtd_benchmark.visualization import ResultAggregator
>>> agg = ResultAggregator(baseline_model="GReaT", dcr_reference=0.42)
>>> agg.add_model("GReaT",    result0=r0a, result1=r1a, result3=r3a, result4=r4a)
>>> agg.add_model("REaLTabF", result0=r0b, result1=r1b, result3=r3b, result4=r4b)
>>> lb = agg.to_leaderboard()
>>> fig_radar   = agg.plot_radar()
>>> fig_trade   = agg.plot_trade_offs()
"""

from llm_gtd_benchmark.visualization.aggregator import ResultAggregator

__all__ = ["ResultAggregator"]

# Analysis — Dataset Profiling & Statistical Significance Testing

`llm_gtd_benchmark.analysis` provides two complementary utilities for
understanding your dataset before benchmarking and for drawing statistically
sound conclusions from benchmark results.

---

## DatasetProfiler

Generates a comprehensive statistical profile of a DataFrame **before**
running the benchmark, helping you understand dataset characteristics
that may affect evaluation (e.g., class imbalance, functional dependencies,
multimodal distributions).

### Usage

```python
from llm_gtd_benchmark.analysis import DatasetProfiler

profiler = DatasetProfiler(top_k_pairs=5)
profile = profiler.profile(df, dataset_name="adult")
print(profile.summary)
```

### DatasetProfile attributes

| Attribute | Type | Description |
|---|---|---|
| `n_rows`, `n_cols` | `int` | Dataset shape |
| `total_missing_rate` | `float` | Overall fraction of missing values |
| `column_profiles` | `Dict[str, ColumnProfile]` | Per-column statistics |
| `top_pearson_pairs` | `List[Tuple[str, str, float]]` | Top-|r| continuous pairs |
| `top_cramer_pairs` | `List[Tuple[str, str, float]]` | Top-V categorical pairs |
| `fd_candidates` | `List[Tuple[str, str, float]]` | Approx. FDs (violation rate < 5%) |
| `class_imbalance_ratio` | `float` | max_freq / min_freq for inferred target |

### ColumnProfile attributes

| Attribute | Continuous | Categorical | Description |
|---|---|---|---|
| `missing_rate` | ✓ | ✓ | Fraction NaN |
| `n_valid` | ✓ | ✓ | Non-null count |
| `mean`, `std`, `min`, `max` | ✓ | — | Basic statistics |
| `skewness`, `kurtosis` | ✓ | — | Third and fourth standardized moments |
| `is_multimodal` | ✓ | — | BC > 5/9 (Pfister et al. 2013) |
| `n_unique` | — | ✓ | Distinct value count |
| `top_value` | — | ✓ | Most frequent category |
| `top_freq_ratio` | — | ✓ | Relative frequency of top value |

### Bimodality Coefficient (BC)

```
BC = (γ₁² + 1) / (γ₂ + 3·(n-1)²/((n-2)·(n-3)))
```

where γ₁ = skewness, γ₂ = excess kurtosis.  BC > 5/9 ≈ 0.556 indicates
likely non-unimodal marginal distribution (Pfister et al. 2013).  This flag
appears in `ColumnProfile.is_multimodal` and is surfaced in `profile.summary`.

### No composite score

`DatasetProfile` provides raw per-dimension statistics; no composite
"difficulty score" is computed. Composite scores introduce arbitrary weighting
that varies across use-cases. Users can combine statistics as needed.

---

## SignificanceTester

Tests whether two models differ significantly across benchmark metrics.
Supports both multi-run and single-run comparison regimes.

### Multi-run comparison (≥ 2 runs per model)

```python
from llm_gtd_benchmark.analysis import SignificanceTester
from llm_gtd_benchmark import BenchmarkPipeline, PipelineConfig

# Run each model 5 times on different synthetic samples
bundles_great = [pipeline_great.run(synth) for synth in synth_runs_great]
bundles_real  = [pipeline_real.run(synth)  for synth in synth_runs_real]

tester = SignificanceTester(alpha=0.05)
report = tester.compare(
    bundles_great,
    bundles_real,
    model_a="GReaT",
    model_b="REaLTabFormer",
    metrics=["mean_ks", "alpha_precision", "dcr_5th_percentile"],
)
print(report.summary)
```

**Test selection:**
- k ≥ 5 matched runs: **Wilcoxon signed-rank** (non-parametric, no normality
  assumption; effect size = rank-biserial correlation).
- 2 ≤ k ≤ 4 matched runs: **Paired t-test** (sensitive to non-normality;
  effect size = Cohen's d).

### Single-run comparison (CI overlap)

When each model is evaluated exactly once but bootstrap CIs were enabled
(`n_boot > 0` in `PipelineConfig`), the tester uses **non-overlapping CIs**
as a significance proxy.

```python
# Both pipelines must have been run with n_boot > 0
bundle_a = BenchmarkPipeline(config_a).run(synth_a)   # n_boot=1000
bundle_b = BenchmarkPipeline(config_b).run(synth_b)   # n_boot=1000

report = tester.compare([bundle_a], [bundle_b], model_a="GReaT", model_b="REaLTabFormer")
```

> **Note:** CI non-overlap implies significance; CI overlap does **not** imply
> non-significance. The single-run test is conservative by design.

### Multiple testing correction

All p-values are corrected with **Holm–Bonferroni** step-down correction
(Holm 1979), which is uniformly more powerful than Bonferroni while
controlling the family-wise error rate at the nominal α.

### Metric registry

The following metrics are pre-registered in `_METRIC_REGISTRY`:

| Metric key | Dimension | Higher = better | CI available |
|---|---|---|---|
| `irr` | 0 | ✓ | — |
| `mean_ks` | 1 | ✗ (lower = better) | ✓ |
| `mean_tvd` | 1 | ✗ | ✓ |
| `alpha_precision` | 1 | ✓ | ✓ |
| `beta_recall` | 1 | ✓ | ✓ |
| `c2st_auc_mean` | 1 | ✗ (0.5 = perfect) | — |
| `pearson_matrix_error` | 1 | ✗ | — |
| `cramerv_matrix_error` | 1 | ✗ | — |
| `icvr` | 2 | ✓ | — |
| `hcs_violation_rate` | 2 | ✗ | — |
| `mdi_mean` | 2 | ✓ | — |
| `dsi_gap` | 2 | ✗ | — |
| `mle_tstr_primary` | 3 | ✓ | — |
| `dcr_5th_percentile` | 4 | ✓ | ✓ |
| `exact_match_rate` | 4 | ✗ | ✓ |
| `delta_eo_mean` | 5 | ✗ | ✓ |
| `delta_dp_mean` | 5 | ✗ | ✓ |

Pass `metrics=None` (default) to test all registered metrics, or provide a
custom list: `metrics=["mean_ks", "alpha_precision", "dcr_5th_percentile"]`.

### SignificanceReport attributes

| Attribute | Description |
|---|---|
| `model_a`, `model_b` | Model names |
| `n_runs_a`, `n_runs_b` | Number of evaluation runs |
| `alpha` | FWER threshold |
| `correction` | Correction method (`"holm"`) |
| `results` | Dict of `MetricTestResult` keyed by metric name |
| `significant_metrics` | Property: list of metrics with significant differences |
| `summary` | Property: formatted comparison table string |

### MetricTestResult attributes

| Attribute | Description |
|---|---|
| `metric` | Metric name |
| `value_a`, `value_b` | Mean scalar values for each model |
| `difference` | `value_b − value_a` |
| `test_method` | `"wilcoxon"`, `"paired_t"`, `"ci_overlap"`, or `"na"` |
| `p_value` | Raw p-value (`NaN` for CI-overlap) |
| `adjusted_p_value` | Holm–Bonferroni corrected p-value |
| `significant` | `True` iff difference is significant at α |
| `effect_size` | Rank-biserial r (Wilcoxon) or Cohen's d (t-test) |
| `ci_overlap` | `True/False` for CI-overlap method; `None` otherwise |

# Pipeline — Unified Benchmark Runner

`llm_gtd_benchmark.pipeline` provides a single entry point for running all six evaluation dimensions in sequence, with automatic fault isolation and structured result serialization.

---

## Quick start

```python
from llm_gtd_benchmark import (
    DataSchema, BenchmarkPipeline, PipelineConfig, TaskType,
)
from llm_gtd_benchmark.core.logic_spec import LogicSpec
from llm_gtd_benchmark.metrics.dimension5 import FairSpec

schema = DataSchema(train_df)

config = PipelineConfig(
    schema=schema,
    train_real_df=train_df,
    test_real_df=test_df,
    model_name="GReaT",
    dataset_name="adult",
    target_col="income",
    task_type=TaskType.BINARY_CLASS,
    fair_spec=FairSpec(["sex", "race"], "income", TaskType.BINARY_CLASS),
    n_boot=1000,          # enable bootstrap CIs for Dim1, Dim4, Dim5
    random_state=42,
    notes="baseline run v1",
)

bundle = BenchmarkPipeline(config).run(synth_df)
bundle.save("results/great_adult.json")

if bundle.has_errors:
    for dim, msg in bundle.errors.items():
        print(f"{dim}: {msg[:200]}")
```

---

## PipelineConfig

| Parameter | Type | Default | Description |
|---|---|---|---|
| `schema` | `DataSchema` | required | Built from real training data |
| `train_real_df` | `pd.DataFrame` | required | Real training set |
| `test_real_df` | `pd.DataFrame` | `None` | Real test set (required for Dim3, Dim5) |
| `model_name` | `str` | required | Generator model identifier |
| `dataset_name` | `str` | required | Dataset identifier |
| `dimensions` | `List[int]` | `[0,1,2,3,4,5]` | Dimensions to evaluate |
| `target_col` | `str` | `None` | Target column (required for Dim3, Dim5) |
| `task_type` | `TaskType` | `None` | Task type (required for Dim3, Dim5) |
| `logic_spec` | `LogicSpec` | `None` | Logic constraints for Dim2 |
| `fair_spec` | `FairSpec` | `None` | Fairness spec for Dim5 |
| `n_boot` | `int` | `0` | Bootstrap replicates for CIs (0 = disabled) |
| `boot_ci` | `float` | `0.95` | Bootstrap confidence level |
| `random_state` | `int` | `42` | Global random seed |
| `notes` | `str` | `""` | Stored in `BundleMetadata.notes` |

---

## Execution model

```
synth_df → [Dim0] ──────────────────────────────────────────────────────→ ResultBundle
               │
               │ clean_df (Dim0 output)
               ├──→ [Dim1]  ←── schema, real_df, (n_boot)
               ├──→ [Dim2]  ←── schema, real_df, logic_spec?
               ├──→ [Dim3]  ←── schema, target_col, task_type, test_df
               ├──→ [Dim4]  ←── schema, real_df, (n_boot)
               └──→ [Dim5]  ←── schema, fair_spec, real_df, test_df, (n_boot)
```

- **Dim0 is mandatory.** If it fails, all downstream dimensions are immediately
  recorded as `"skipped: Dim0 failed"` and the bundle is returned.
- **Dim1–5 are independent.** A failure in Dim2 does not prevent Dim3–5 from
  running; each dimension catches its own exceptions.
- **Configuration omissions** (e.g., Dim3 without `target_col`) are recorded as
  `"skipped: …"` entries in `bundle.errors` — distinct from runtime exceptions.

---

## Error inspection

```python
for dim, msg in bundle.errors.items():
    if msg.startswith("skipped: "):
        print(f"{dim} was intentionally skipped — {msg[9:]}")
    else:
        print(f"{dim} FAILED:\n{msg}")  # full traceback
```

---

## ResultBundle — serialization

The pipeline returns a `ResultBundle` that can be saved to and loaded from a
single JSON file, enabling experiment tracking across runs.

```python
# Save
bundle.save("results/great_adult.json")

# Load
from llm_gtd_benchmark import ResultBundle
bundle = ResultBundle.load("results/great_adult.json")

# Validate schema hasn't changed since bundle was created
ok = bundle.validate_schema(schema)   # False → schema drift warning

# Inspect
print(bundle.dimensions_computed)     # ['dim0', 'dim1', 'dim2', 'dim4']
print(bundle.has_errors)              # True / False
print(bundle.metadata.timestamp)      # ISO-8601 UTC string
```

### Serialization notes

- `NaN` floats → JSON `null`, and back to `float("nan")` on load.
- `Dim0Result.clean_df` is **not** serialized (too large). On load it is an
  empty DataFrame; re-run `StructuralInterceptor` to repopulate it.
- Bootstrap CI tuples `(lo, hi)` → JSON `[lo, hi]` or `null`.
- `format_version` is stored; a version mismatch on load emits a warning
  but does **not** raise (forward-compatible via unknown-field-ignore policy).

---

## Bootstrap CIs

When `n_boot > 0`, per-sample bootstrap confidence intervals are computed for
the following metrics:

| Metric | Dimension | Method |
|---|---|---|
| `mean_ks_ci` | Dim1 | Bootstrap mean of per-column KS scores |
| `mean_tvd_ci` | Dim1 | Bootstrap mean of per-column TVD scores |
| `alpha_precision_ci` | Dim1 | Bootstrap proportion of per-synth-point authenticity indicators |
| `beta_recall_ci` | Dim1 | Bootstrap proportion of per-real-point coverage indicators |
| `dcr_5th_ci` | Dim4 | Bootstrap 5th percentile of per-row DCR distances |
| `exact_match_rate_ci` | Dim4 | Bootstrap proportion of exact-match boolean indicators |
| `delta_eo_ci` | Dim5 | Bootstrap row-triplet resampling of ΔEO per protected column |
| `delta_dp_ci` | Dim5 | Bootstrap row-triplet resampling of ΔDP per protected column |

These CIs measure **sampling uncertainty** of the metrics given the finite
test set — not variance across multiple generator runs.  For inter-model
comparison with uncertainty estimates, see
[`SignificanceTester`](../analysis/README.md).

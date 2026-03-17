# `utils/` — 共享工程基础设施层

本模块提供被多个 metrics 模块复用的底层工具，**不包含任何业务逻辑**。

---

## `nn_backend.py` — 近邻索引抽象层

### 设计动机

Dimension 1 的 α-precision / β-recall 需要大规模 KNN 搜索。`faiss` 比 sklearn 快 10~100 倍，但在 Windows 上 pip 安装不稳定。本模块通过抽象层解耦性能与兼容性：

```
NNIndex（抽象接口）
├── SklearnNNIndex  — 永远可用，基于 BallTree/KDTree
└── FaissNNIndex    — 可选，基于 faiss IndexFlatL2
```

### 使用方式

```python
from llm_gtd_benchmark.utils import build_nn_index

index = build_nn_index(data_array)              # 自动选最优后端
index = build_nn_index(data_array, force_backend="faiss")   # 强制 faiss
index = build_nn_index(data_array, force_backend="sklearn")  # 强制 sklearn

distances, indices = index.query(query_array, k=5)
# distances: shape (n_queries, 5), L2 欧氏距离（非平方）
# indices:   shape (n_queries, 5), 参考集行索引
```

### 自动选择逻辑

| 条件 | 选择 |
|---|---|
| faiss 不可用 | sklearn（必然 fallback） |
| n_samples < 10,000 | sklearn（小数据不值得用 faiss） |
| n_samples ≥ 10,000 且 faiss 可用 | faiss |
| `force_backend="faiss"` 且 faiss 未安装 | 抛出 `ImportError`（明确失败） |

---

## `preprocessing.py` — 特征编码 + OOM 安全子采样

### `build_feature_encoder(schema, real_df) → ColumnTransformer`

在 `real_df` 上 fit 一个 sklearn ColumnTransformer：
- 连续列 → `StandardScaler`（零均值，单位方差）
- 离散列 → `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

**关键设计：只 fit 一次，多次 transform。** Dimension 1 的所有子指标（KNN、C2ST）共用同一个 encoder，保证编码空间一致性。对 OOV 类别产生全零行（中性编码，不膨胀距离）。

### `stratified_subsample(df, max_rows, strat_col=None, random_state=42)`

OOM 防护：数据超过 `max_rows`（默认 50,000）时进行子采样。

| 策略 | 条件 |
|---|---|
| 分层采样（`train_test_split`） | `strat_col` 指定且类别够大 |
| 均匀随机采样 | 分层失败或未指定 `strat_col` |
| 直接返回（无 copy） | `len(df) ≤ max_rows` |

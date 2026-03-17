# `core/` — 基础数据契约层

本模块是整个包的"宪法"层。所有上层模块（metrics、utils）只依赖这里定义的类和异常，不依赖彼此，保证了单向依赖图。

---

## 文件一览

### `schema.py` — 统计元数据

```python
DataSchema(real_df, categorical_columns=None, continuous_columns=None)
```

从真实训练数据**一次性推断**所有列的统计元数据，之后**不可变（frozen）**。

| 字段 | 说明 |
|---|---|
| `col_type` | `"continuous"` 或 `"categorical"`（自动检测 + 手动覆盖） |
| `dtype` | 原始 numpy dtype（用于 Dim0 强制下转型） |
| `min_val / max_val` | 连续列的训练集域边界（用于 Dim0 OOB 检查） |
| `categories` | 离散列的封闭词表 frozenset（用于 Dim0 OOV 检查） |

**自动检测规则：** `object`/`bool`/`category` dtype → categorical；unique 数 ≤ 20 的数值列 → categorical；其余 → continuous。阈值可通过 `categorical_threshold` 参数调整。

---

### `logic_spec.py` — 语义约束元数据

```python
LogicSpec(name, known_fds=[], hierarchies=[], math_equations=[])
discover_fds(real_df, schema)  # 自动扫描候选 FD
```

捕捉数据集的**领域语义先验**，供 Dimension 2 路由使用。与 `DataSchema` 的区别：

| | DataSchema | LogicSpec |
|---|---|---|
| 内容 | 统计元数据（类型、边界、词表） | 逻辑约束（FD、层次树、数学恒等式） |
| 构建方式 | 自动从 real_df 推断 | 手动注册（+ `discover_fds` 辅助） |
| 必要性 | 所有维度都必须 | 仅 Dim2 的 ICVR/HCS/MDI 需要 |

---

### `exceptions.py` — 异常层级

```
BenchmarkError（基类）
├── GenerationCollapseError   — 有效行数不足阈值（模型灾难性失败）
├── SchemaMismatchError       — 合成数据列集合与 Schema 不匹配
└── InsufficientDataError     — 某指标的样本量不足以可靠计算
```

所有包异常都可用 `except BenchmarkError` 统一捕获。

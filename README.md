1# LLM-GTD-Benchmark

**A rigorous, modular benchmark suite for evaluating LLM-based tabular data generation models.**

> 现有开源库（GReaT、REaLTabFormer 等）默认生成数据是"干净矩阵"。本包的核心差异化在于：将"脏数据清洗"提升为一阶评估指标，引入完整的多维度保真度闭环，并通过自动化逻辑探针支持任意数据集的无代码扩展。

---

## 安装

```bash
# 基础安装
pip install -e .

# 含 faiss 加速（Linux/macOS，大数据集 KNN 快 10~100 倍）
pip install -e ".[faiss]"

# 开发依赖
pip install -e ".[dev]"
```

**核心依赖：** `numpy >= 1.24` · `pandas >= 2.0` · `scipy >= 1.11` · `scikit-learn >= 1.3` · `xgboost >= 1.7`

---

## 评估维度总览

| 维度 | 模块 | 指标 | 适用性 |
|---|---|---|---|
| **0** 结构合法性 | `StructuralInterceptor` | IRR（废片率）、缺陷分布 | 全局 |
| **1** 分布保真度 | `FidelityEvaluator` | KS、TVD、Pearson、CramerV、α-precision、β-recall、C2ST AUC | 全局 |
| **2** 逻辑约束 | `LogicEvaluator` | DSI（GMM对数似然）、ICVR（FD违背率）、HCS（层次违背率）、MDI（算术恒等式违背率） | DSI全局；ICVR/HCS/MDI按注册 |
| **3** 下游任务效用 | `MLUtilityEvaluator` / `LLMUtilityEvaluator` | TSTR/TRTR ROC-AUC、Macro-F1（分类）；R²、WMAPE（回归）；LLE Macro-F1（可选） | 全局（需指定 target_col） |
| **4** 隐私与防记忆 | `PrivacyEvaluator` / `MemorizationProbe` | DCR 5th percentile、Exact Match Rate；Masked DLT PPL gap（可选） | 黑盒全局；白盒需生成器权重 |
| **5** 公平性与去偏 | `FairnessEvaluator` | NMI（内在偏差）、ΔDP、ΔEOP、ΔEO（分类）；ΔSPG（回归）；可选交叉公平性 | 全局（需指定 protected_cols） |

---

## 快速开始

### 最简流程

```python
import pandas as pd
from llm_gtd_benchmark import DataSchema, StructuralInterceptor, FidelityEvaluator

# 1. 构建 Schema（从真实训练集）
real_df = pd.read_csv("real_train.csv")
schema = DataSchema(real_df)
print(schema)
# DataSchema(8 columns: 5 continuous, 3 categorical)

# 2. 维度 0：结构拦截（输入为模型生成的 CSV，全部以字符串读入）
synth_df = pd.read_csv("synth_output.csv", dtype=str)
result0 = StructuralInterceptor(schema).evaluate(synth_df)
print(result0.summary)
# ── Dimension 0: Structural Validity ───────────────────────
#   Total generated rows :    10000
#   Invalid rows         :      312  (3.12%)
#   Clean rows           :     9688
#   Defect breakdown:
#     type_coercion_nan  :      102
#     oov_hallucination  :      178
#     out_of_bounds      :       32

# 3. 维度 1：分布保真度（输入 clean_df）
result1 = FidelityEvaluator(schema, real_df).evaluate(result0.clean_df)
print(result1.summary)
```

---

### 含逻辑约束（维度 2）

```python
from llm_gtd_benchmark import LogicSpec, LogicEvaluator, discover_fds

# 先用自动发现辅助工具找候选 FD
candidates = discover_fds(real_df, schema)
print(candidates)
# [('education_num', 'education'), ('zip_code', 'city'), ...]

# 注册已确认的约束
spec = LogicSpec(
    name="adult_income",
    known_fds=[
        ("education_num", "education"),   # 教育年限→教育程度（1-to-1）
    ],
    hierarchies=[
        ["country", "state", "city"],     # 国家→州→城市层次树
    ],
    math_equations=[
        ("unit_price", "*", "quantity", "total_price"),  # 单价×数量=总价
    ],
)

result2 = LogicEvaluator(schema, real_df, logic_spec=spec).evaluate(result0.clean_df)
print(result2.summary)
# ── Dimension 2: Cross-Column Logic & Dependencies ────────
#   DSI (Distributional Similarity Index):
#     Synth log-likelihood        : -3.5375
#     Real  log-likelihood (ref)  : -3.5172
#     Gap   (↓ better, 0=perfect) : 0.0204
#     GMM components selected     : 3
#   ICVR (Functional Dependency Violation Rate, ↓ better):
#     education_num→education     : 0.0023
#     Overall ICVR               : 0.0023
#   HCS (Hierarchy Violation Rate, ↓ better):
#     country→state→city          : 0.0041
#   MDI (Arithmetic Violation Rate, ↓ better):
#     unit_price * quantity = total_price: 0.0187
```

---

### Schema 手动指定列类型

```python
# 覆盖自动检测：强制 zip_code 为 categorical（否则会被检测为 continuous）
schema = DataSchema(
    real_df,
    categorical_columns=["zip_code", "education_num"],
    continuous_columns=["income"],
    categorical_threshold=20,  # 自动检测阈值，默认 20
)
```

---

### 自定义 KNN 后端 / 参数

```python
from llm_gtd_benchmark import FidelityEvaluator

evaluator = FidelityEvaluator(
    schema, real_df,
    n_neighbors=10,          # k-NN 邻居数，默认 5
    max_samples=30_000,      # OOM 子采样上限，默认 50,000
    c2st_n_splits=5,         # C2ST 交叉验证折数
    random_state=0,          # 复现种子
    nn_backend="sklearn",    # 强制 sklearn（faiss 未安装时的显式指定）
    c2st_strat_col="label",  # C2ST 子采样的分层列
)
result = evaluator.evaluate(clean_df)
```

---

## 项目结构

```
llm_gtd_benchmark/
├── __init__.py                  ← 公共 API 入口
├── core/
│   ├── schema.py                ← DataSchema（统计元数据，不可变）
│   ├── logic_spec.py            ← LogicSpec + discover_fds（语义约束）
│   ├── result_bundle.py         ← ResultBundle / BundleMetadata（JSON 序列化）
│   └── exceptions.py            ← BenchmarkError 异常层级
├── metrics/
│   ├── dimension0.py            ← StructuralInterceptor：IRR + 干净数据
│   ├── dimension1.py            ← FidelityEvaluator：8 个保真度指标 + Bootstrap CI
│   ├── dimension2.py            ← LogicEvaluator：DSI / ICVR / HCS / MDI
│   ├── dimension3.py            ← MLUtilityEvaluator + LLMUtilityEvaluator：TSTR 效用
│   ├── dimension4.py            ← PrivacyEvaluator + MemorizationProbe：隐私 / DCR / DLT + Bootstrap CI
│   └── dimension5.py            ← FairnessEvaluator：NMI 偏差 + TSTR 探针差异指标 + Bootstrap CI
├── pipeline/
│   ├── config.py                ← PipelineConfig 数据类
│   └── runner.py                ← BenchmarkPipeline（顺序执行 + 故障隔离）
├── analysis/
│   ├── profiler.py              ← DatasetProfiler：列级统计 + 关联对 + FD 候选
│   └── significance.py          ← SignificanceTester：Wilcoxon / 配对 t / CI 不重叠 + Holm 校正
├── visualization/
│   └── aggregator.py            ← ResultAggregator：雷达图 / 排行榜 / Pareto 前沿
└── utils/
    ├── bootstrap.py             ← Bootstrap CI 工具函数
    ├── nn_backend.py            ← NNIndex 抽象（faiss / sklearn 自动切换）
    └── preprocessing.py         ← 特征编码器 + OOM 安全子采样
```

---

## 指标速查表

### 维度 0

| 指标 | 范围 | 越好 |
|---|---|---|
| IRR（Invalid Row Rate） | [0, 1] | ↓ 越低越好 |

### 维度 1

| 指标 | 范围 | 越好 |
|---|---|---|
| KS statistic（per column + mean） | [0, 1] | ↓ |
| TVD（per column + mean） | [0, 1] | ↓ |
| Pearson matrix error | [0, 2] | ↓ |
| Cramér's V matrix error（偏差校正） | [0, 1] | ↓ |
| α-precision | [0, 1] | ↑ |
| β-recall | [0, 1] | ↑ |
| C2ST AUC（XGBoost + RF） | [0.5, 1] | → 0.5 |

### 维度 2

| 指标 | 范围 | 越好 | 需要注册 |
|---|---|---|---|
| DSI gap（real_ll − synth_ll） | ≥ 0 | ↓（0=完美） | 无 |
| DSI synth_ll | 无界 | ↑ | 无 |
| ICVR | [0, 1] | ↓ | `known_fds` |
| HCS violation rate | [0, 1] | ↓ | `hierarchies` |
| MDI violation rate | [0, 1] | ↓ | `math_equations` |

### 维度 3

| 指标 | 范围 | 越好 | 备注 |
|---|---|---|---|
| ROC-AUC (TSTR / TRTR / gap) | [0, 1] | ↑ / gap→0 | 分类任务 |
| Macro-F1 (TSTR / TRTR / gap) | [0, 1] | ↑ / gap→0 | 分类任务 |
| R² (TSTR / TRTR / gap) | (−∞, 1] | ↑ / gap→0 | 回归任务 |
| WMAPE (TSTR / TRTR) | ≥ 0 | ↓ | 回归任务 |
| LLE Macro-F1 | [0, 1] | ↑ | 可选，需 GPU |

### 维度 4

| 指标 | 范围 | 越好 | 备注 |
|---|---|---|---|
| DCR 5th percentile | ≥ 0 | ↑ | 黑盒，无额外依赖 |
| Exact match rate | [0, 1] | ↓ | > 1% 触发 DataCopyingWarning |
| Masked DLT gap | 无界 | → 0 | 白盒，需生成器权重 |

### 维度 5

| 指标 | 范围 | 越好 | 备注 |
|---|---|---|---|
| NMI_real（参考基线） | [0, 1] | ↓ | 真实数据偏差基线 |
| NMI_synth（审计对象） | [0, 1] | ↓ | 正差 = 放大偏差；负差 = 抑制偏差 |
| ΔDP（Demographic Parity） | [0, 1] | ↓ | 分类任务；max 组间正例预测率差 |
| ΔEOP（Equal Opportunity） | [0, 1] | ↓ | 分类任务；max 组间 TPR 差 |
| ΔEO（Equalized Odds） | [0, 1] | ↓ | 分类任务；max(TPR 差, FPR 差) |
| ΔSPG（Statistical Parity Gap） | ≥ 0 | ↓ | 回归任务；max 组间预测均值差 |
| 交叉 ΔDP | [0, 1] | ↓ | intersectional=True 时计算 |

---

### 公平性评估（维度 5）

```python
from llm_gtd_benchmark import FairSpec, FairnessEvaluator, TaskType

# 定义公平性规格（纯规格对象，不含数据）
spec = FairSpec(
    protected_cols=["gender", "race"],
    target_col="income",
    task_type=TaskType.BINARY_CLASS,
    intersectional=True,   # 开启笛卡尔积交叉公平性
    min_group_size=30,     # 少于 30 样本的分组触发 GroupCollapseWarning
)

# 初始化评估器（__init__ 中 fit KBinsDiscretizer，锚定到真实训练集）
evaluator = FairnessEvaluator(schema, spec, train_real_df)

# 评估（synth_df 用于训练内部 XGBoost 探针，test_real_df 用于分组评测）
result5 = evaluator.evaluate(synth_df=result0.clean_df, test_real_df=test_real_df)

print(result5.summary)
# ── Dimension 5: Fairness & Debiasing ───────────────────────────────
#   Target    : income
#   Task      : binary_classification
#   Protected : gender, race
#
#   Intrinsic Bias — NMI(A; Y)  [↓ less bias | 0 = independent]
#   Attribute                NMI_real (ref)     NMI_synth          Δ(synth−real)
#   ────────────────────────────────────────────────────────────────────────────
#   gender                   0.0821             0.0854             0.0033   (↑ amplified)
#   race                     0.1103             0.1056            -0.0047   (↓ suppressed)
#
#   Downstream Disparity — Classification  [↓ fairer | 0 = perfect parity]
#   Attribute                ΔDP          ΔEOP         ΔEO
#   ────────────────────────────────────────────────────────
#   gender                   0.0412       0.0318       0.0487
#   race                     0.0731       0.0623       0.0814
#
#   Intersectional ΔDP : 0.1192
```

---

## 多模型对比可视化

`ResultAggregator` 将多个模型的评估结果聚合为三种出版级图表，以及含 Bootstrap 置信区间的排行榜 DataFrame。

### 快速示例

```python
from llm_gtd_benchmark.visualization import ResultAggregator

# dcr_reference：真实数据留存集的自 DCR（用于 Privacy 轴绝对归一化）
agg = ResultAggregator(baseline_model="GReaT", dcr_reference=0.42)

# 每次 add_model 代表一个评估运行；同名多次调用 → Bootstrap CI
agg.add_model("GReaT",     result0=r0a, result1=r1a, result2=r2a,
                            result3=r3a, result4=r4a, result5=r5a)
agg.add_model("REaLTabF",  result0=r0b, result1=r1b, result2=r2b,
                            result3=r3b, result4=r4b, result5=r5b)
agg.add_model("TabLLM",    result0=r0c, result1=r1c, result2=r2c,
                            result3=r3c, result4=r4c, result5=r5c)

# 1. 排行榜（含 Bootstrap CI，n_boot=1000）
lb = agg.to_leaderboard(n_boot=1000, ci=0.95)
print(lb[["composite_score", "composite_ci_lo", "composite_ci_hi",
          "utility_tstr_auc", "privacy_dcr_p5", "fairness_delta_eo_mean"]])

# 2. 六维雷达图
fig_radar = agg.plot_radar(normalize="baseline")   # 相对基线归一化
fig_radar.savefig("radar.pdf", bbox_inches="tight")

# 3. 三联 Pareto 前沿图（效用×隐私 / 效用×公平 / 隐私×公平）
fig_trade = agg.plot_trade_offs()
fig_trade.savefig("trade_offs.pdf", bbox_inches="tight")
```

### 图表说明

| 图表 | 方法 | 说明 |
|---|---|---|
| **雷达图** | `plot_radar()` | 六轴极坐标图，每轴对应一个评估维度；支持 `"minmax"` 或 `"baseline"` 归一化 |
| **排行榜** | `to_leaderboard()` | 所有关键指标 + 加权综合分 + Bootstrap CI；`compact=True` 输出 7 列摘要 |
| **单 Pareto** | `plot_pareto(x, y)` | 任意两维度的 Pareto 前沿散点图；Pareto 最优点标注 ★ |
| **三联 Pareto** | `plot_trade_offs()` | A: 效用×隐私；B: 效用×公平；C: 隐私×公平（揭示 Pujol et al. 2020 中的隐私-公平张力） |

### 默认综合权重

| 维度 | 权重 |
|---|---|
| 结构合法性（Dim0） | 0.15 |
| 分布保真度（Dim1） | 0.20 |
| 逻辑约束（Dim2） | 0.20 |
| 下游效用（Dim3） | 0.20 |
| 隐私（Dim4） | 0.15 |
| 公平性（Dim5） | 0.10 |

权重可通过 `ResultAggregator(composite_weights={...})` 自定义（需合计为 1.0）。

---

## 统一实验 Pipeline

`BenchmarkPipeline` 提供一键运行全部六个维度的接口，自动隔离各维度故障，并将结果序列化为单一 JSON 文件。

```python
from llm_gtd_benchmark import BenchmarkPipeline, PipelineConfig, TaskType
from llm_gtd_benchmark.metrics.dimension5 import FairSpec

config = PipelineConfig(
    schema=schema,
    train_real_df=train_df,
    test_real_df=test_df,
    model_name="GReaT",
    dataset_name="adult",
    target_col="income",
    task_type=TaskType.BINARY_CLASS,
    fair_spec=FairSpec(["sex", "race"], "income", TaskType.BINARY_CLASS),
    n_boot=1000,       # 启用 Bootstrap CI（Dim1 / Dim4 / Dim5）
    random_state=42,
)

bundle = BenchmarkPipeline(config).run(synth_df)
bundle.save("results/great_adult.json")

# 检查失败原因
for dim, msg in bundle.errors.items():
    print(f"{dim}: {msg[:120]}")
```

**故障隔离规则：**
- Dim0 失败 → 全部下游标记 `"skipped: Dim0 failed"` 并立即返回
- Dim1–5 相互独立，单维度失败不影响其他维度
- 配置缺失（如没有 `target_col`）→ `"skipped: ..."` 前缀，区别于运行时异常

详见 [`pipeline/README.md`](llm_gtd_benchmark/pipeline/README.md)。

---

## 结果序列化（ResultBundle）

`ResultBundle` 支持将完整评估结果（含 Bootstrap CI）保存为单一 JSON 文件，便于实验追踪。

```python
from llm_gtd_benchmark import ResultBundle

# 保存
bundle.save("results/great_adult.json")

# 加载
bundle = ResultBundle.load("results/great_adult.json")
print(bundle.metadata.timestamp)        # ISO-8601 UTC 时间戳
print(bundle.dimensions_computed)       # ['dim0', 'dim1', 'dim2', 'dim4']
print(bundle.has_errors)                # True / False

# Schema 漂移检测
ok = bundle.validate_schema(schema)     # False → 发出警告
```

序列化约定：NaN ↔ JSON `null`；Bootstrap CI 元组 ↔ `[lo, hi]`；`Dim0.clean_df` 不序列化（体积过大）。

---

## 数据集自动画像

`DatasetProfiler` 在基准测试前提供数据集统计概览，帮助理解影响评估的关键特征。

```python
from llm_gtd_benchmark.analysis import DatasetProfiler

profiler = DatasetProfiler(top_k_pairs=5)
profile = profiler.profile(real_df, dataset_name="adult")
print(profile.summary)
# ── Dataset Profile: adult ──────────────────────────────────────────
#   Rows: 48,842   Cols: 15   Overall missing: 0.73%
#   Continuous: 6   Categorical: 9
#   Possibly multimodal (BC > 5/9): age, hours-per-week
#   Top Pearson pairs:
#     education-num × education: r = +0.974
#   Top Cramér's V pairs:
#     sex × relationship: V = 0.658
#   Approx. functional dependencies (violation rate < 5 %):
#     education-num → education  (violation rate: 0.00%)
#   Class imbalance ratio (max/min freq): 3.17
```

详见 [`analysis/README.md`](llm_gtd_benchmark/analysis/README.md)。

---

## 指标显著性检验

`SignificanceTester` 对多模型评估结果进行统计显著性检验，支持多次运行（Wilcoxon / 配对 t 检验）和单次运行（Bootstrap CI 不重叠代理检验）两种场景，全部应用 Holm–Bonferroni 多重比较校正。

```python
from llm_gtd_benchmark.analysis import SignificanceTester

tester = SignificanceTester(alpha=0.05)
report = tester.compare(
    bundles_great,      # 5 次运行的 ResultBundle 列表
    bundles_real,
    model_a="GReaT",
    model_b="REaLTabFormer",
    metrics=["mean_ks", "alpha_precision", "dcr_5th_percentile", "delta_eo_mean"],
)
print(report.summary)
# ── Significance Report: GReaT vs. REaLTabFormer ─────────────────────
#   Runs: A=5, B=5  |  α=0.05  |  correction=holm
#
#   Metric                       A          B          Diff    Method         adj-p  Sig?  Effect
#   ─────────────────────────────────────────────────────────────────────────────────────────────
#   mean_ks                  0.1023     0.0871     -0.0152  wilcoxon          0.038  ✓       0.720
#   alpha_precision          0.8341     0.8712     +0.0371  wilcoxon          0.062
#   dcr_5th_percentile       0.2145     0.2389     +0.0244  wilcoxon          0.125
#   delta_eo_mean            0.0512     0.0438     -0.0074  wilcoxon          0.250
#
#   Significant metrics (1): mean_ks
```

详见 [`analysis/README.md`](llm_gtd_benchmark/analysis/README.md)。

---

## 工程特性

- **零硬编码数据集依赖**：通过 `DataSchema` + `LogicSpec` + `FairSpec` 支持任意表格数据集
- **OOM 安全**：超 50k 行自动分层子采样；KNN 使用 faiss/sklearn 自适应后端；DCR 双重分块广播
- **FutureWarning 清洁**：所有 pandas 操作经过验证，无弃用警告
- **NaN 优雅降级**：指标不适用时返回 NaN，不抛出异常
- **可复现**：所有随机操作统一接受 `random_state` 参数
- **可选依赖分层**：核心评估（Dim0-2 + Dim4 黑盒 + Dim5）仅需 numpy/pandas/scipy/sklearn；xgboost 可选（Dim3/Dim5 探针 fallback 到 RandomForest）；faiss 可选加速；transformers/peft/trl 仅 Dim3 LLE 和 Dim4 DLT 需要
- **隐私硬护栏**：`DataCopyingWarning` 在检测到近似复制时自动触发
- **公平性硬护栏**：`GroupCollapseWarning` 在保护属性分组样本不足时触发，NaN 替代平滑值防止假性公平读数
- **ΔEOP vs ΔEO 严格区分**（Hardt et al. 2016）：ΔEOP 仅测 TPR（受益侧）；ΔEO 额外测 FPR（伤害侧），防止在信贷等场景下低估对少数群体的假阳性伤害
- **Python 3.9+**，sklearn ≥ 1.2

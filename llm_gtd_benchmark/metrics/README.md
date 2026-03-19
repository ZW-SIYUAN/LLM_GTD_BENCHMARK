# `metrics/` — 评估维度模块

每个维度独立实现，通过统一的 `evaluate(df) → DimNResult` 接口调用。维度间数据流：

```
raw synth_df
     │
     ▼ dimension0.py
clean_df + IRR
     │
     ├──▶ dimension1.py → Dim1Result（保真度）
     ├──▶ dimension2.py → Dim2Result（逻辑约束）
     ├──▶ dimension3.py → Dim3Result（下游任务效用）
     │                    ├─ MLUtilityEvaluator  (TSTR / TRTR + 超参调优)
     │                    └─ LLMUtilityEvaluator (LoRA 微调探针, 可选)
     ├──▶ dimension4.py → Dim4Result（隐私与防记忆）
     │                    ├─ PrivacyEvaluator    (DCR 黑盒距离)
     │                    └─ MemorizationProbe   (Masked DLT 白盒, 可选)
     └──▶ dimension5.py → Dim5Result（公平性与去偏）
                          └─ FairnessEvaluator   (NMI 内在偏差 + TSTR 探针下游差异)

（多模型对比时，将各 DimNResult 传入 visualization/aggregator.py）
     │
     ▼ ResultAggregator.add_model(name, result0-5)
     │
     ├──▶ to_leaderboard()  → pd.DataFrame（排行榜 + Bootstrap CI）
     ├──▶ plot_radar()       → Figure（六维雷达图）
     └──▶ plot_trade_offs()  → Figure（三联 Pareto 前沿图）
```

---

## `dimension0.py` — 结构合法性拦截器

**理论定位：** 概率支撑集投影。在计算任何统计散度之前，强制将 LLM 溢出真实流形的数据点过滤掉，保证后续计算的数学合法性。

**核心类：** `StructuralInterceptor(schema, min_clean_rows=100)`

**核心输出：** `Dim0Result`

| 指标 | 含义 | 方向 |
|---|---|---|
| `irr` | Invalid Row Rate（废片率）[0,1] | 越低越好 |
| `defect_counts["type_coercion_nan"]` | 类型不可解析的行数 | — |
| `defect_counts["oov_hallucination"]` | 词表外幻觉的行数 | — |
| `defect_counts["out_of_bounds"]` | 超出训练集域边界的行数 | — |
| `clean_df` | 通过所有检查的干净 DataFrame | — |

**各指标计算方式：**

- **`irr`**：`n_invalid / n_total`，其中 `n_invalid` 为以下三类缺陷的并集（一行触发任意一类即为无效行）。

- **`type_coercion_nan`**：对每个连续列，尝试将合成值强制转换为 float；转换失败（NaN）的行计入此类。

- **`oov_hallucination`**：对每个分类列，检查合成值是否在训练集词表（`frozenset`）内；不在词表内的视为"词表外幻觉"。特殊情况：若真实列全为 NaN（词表为空集），则合成列中任何非 NaN 值均视为幻觉，NaN 值视为合法。

- **`out_of_bounds`**：对每个连续列，检查合成值是否超出训练集的 `[min, max]` 范围（允许 Schema 中配置的 `bounds_slack` 弹性余量）。

- **`clean_df`**：移除所有无效行后的剩余 DataFrame，直接传入 Dim1–Dim5。

---

## `dimension1.py` — 分布保真度评估器

**理论定位：** 多尺度度量闭环。从 1D 边缘分布到高维拓扑流形再到全局非线性边界，形成严密的保真度测量体系。

**核心类：** `FidelityEvaluator(schema, real_df, n_neighbors=5, max_samples=50_000)`

**核心输出：** `Dim1Result`

| 指标 | 层次 | 含义 | 方向 |
|---|---|---|---|
| `mean_ks` | 低阶 | 连续列 KS 统计量均值 [0,1] | ↓ |
| `mean_tvd` | 低阶 | 离散列总变差距离均值 [0,1] | ↓ |
| `pearson_matrix_error` | 中阶 | Pearson 相关矩阵 MAE | ↓ |
| `cramerv_matrix_error` | 中阶 | Cramér's V 矩阵 MAE（偏差校正） | ↓ |
| `alpha_precision` | 高阶 | 合成点落在真实 k-NN 支撑集内的比例 [0,1] | ↑ |
| `beta_recall` | 高阶 | 真实支撑集被合成点覆盖的比例 [0,1] | ↑ |
| `c2st_auc_xgb` | 全局 | XGBoost 鉴别器 AUC（50% = 完美） | → 0.5 |
| `c2st_auc_rf` | 全局 | RandomForest 鉴别器 AUC | → 0.5 |
| `skipped_columns` | 诊断 | 被排除出边缘分布计算的列及原因（空字典表示所有列均正常） | — |

**各指标计算方式：**

- **`mean_ks`**：对每个连续列调用 `scipy.stats.ks_2samp(real_col, synth_col)`，取返回的 KS 统计量（两条经验 CDF 之间的最大绝对差）；对所有连续列求均值得到 `mean_ks`。

- **`mean_tvd`**：对每个分类列，统计真实与合成的类别频率，计算 `0.5 × Σ|p_i − q_i|`（仅出现在一方的类别按 0 处理）；对所有分类列求均值得到 `mean_tvd`。

- **`pearson_matrix_error`**：对所有连续列分别计算真实和合成的 Pearson 相关矩阵，取上三角（不含对角线）的逐元素绝对差均值（MAE）。

- **`cramerv_matrix_error`**：对所有分类列的每对组合，用 `chi2_contingency` 计算偏差校正后的 Cramér's V，构成相关矩阵；真实和合成矩阵上三角的 MAE 即为该指标。单列无变化（退化列联表）时 Cramér's V 返回 0.0。

- **`alpha_precision` / `beta_recall`**（Alaa et al. 2022）：
  1. 用共享 ColumnTransformer（连续列 StandardScaler + SimpleImputer，分类列 OrdinalEncoder）将真实和合成数据编码为数值矩阵 R 和 S。
  2. 对 R 中每个点，计算其第 k 个真实近邻距离 δ_k(r) 作为"支撑半径"（k 默认 5）。
  3. **α-precision**：对 S 中每个合成点 s，找其最近真实点 r*；若 `d(s, r*) ≤ δ_k(r*)`，则 s 落在真实流形内。α-precision = 满足条件的合成点比例。低 α 说明合成数据偏离真实流形（幻觉多）。
  4. **β-recall**：对 R 中每个真实点 r，找其最近合成点 s*；若 `d(r, s*) ≤ δ_k(r)`，则 r 被合成数据覆盖。β-recall = 满足条件的真实点比例。低 β 说明模式崩溃（合成数据覆盖不全）。

- **`c2st_auc_xgb` / `c2st_auc_rf`**：
  1. 真实数据打标签 `y=1`，合成数据打标签 `y=0`，拼接为混合数据集。
  2. 用同一 ColumnTransformer 编码为数值矩阵。
  3. StratifiedKFold 5 折交叉验证：每折用 4 份训练分类器，在第 5 份（含真实+合成）上用预测概率计算 ROC-AUC；取 5 折均值。
  4. XGBoost 参数：n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8。RandomForest 参数：n_estimators=300, max_depth=10。两者独立运行，分别输出。

**α/β 定义（Alaa et al. 2022）：** 每个真实点的第 k 个近邻距离作为自适应半径，无需人工设定带宽。

**防御机制：** 若某列在真实数据或合成数据中全为 NaN，该列会被跳过（soft skip），不影响其他列的计算。具体原因记录在 `Dim1Result.skipped_columns` 中，同时触发 `logger.warning`，使 `result.summary` 可直接展示跳过详情，便于诊断数据质量问题。

---

## `dimension2.py` — 跨列逻辑约束评估器

**理论定位：** 自动化逻辑探针与路由注册表。通过 `LogicSpec` 元数据驱动，自动跳过不适用的指标（返回 NaN）而非报错，实现对任意数据集的通用评估。

**核心类：** `LogicEvaluator(schema, real_df, logic_spec=None)`

**核心输出：** `Dim2Result`

| 指标 | 适用性 | 含义 | 方向 |
|---|---|---|---|
| `dsi_gap` | 全局（Universal） | GMM 对数似然差距（实际 - 合成），绝对值 | ↓（0 = 完美） |
| `dsi_relative_gap` | 全局 | `dsi_gap / |dsi_real_ll| × 100%`，相对百分比，跨数据集可比 | ↓（0% = 完美） |
| `dsi_synth_ll` | 全局 | 合成数据在真实 GMM 下的对数似然均值 | ↑ |
| `icvr` | 需 `known_fds` | 函数依赖违背率 [0,1] | ↓ |
| `hcs_violation_rate` | 需 `hierarchies` | 层次树拓扑违背率 [0,1] | ↓ |
| `mdi_mean` | 需 `math_equations` | 数学恒等式违背率均值（相对误差 > ε） | ↓ |

**各指标计算方式：**

- **`dsi_gap` / `dsi_synth_ll` / `dsi_relative_gap`**：
  1. 仅取连续列；若连续列数 > 20，先用 PCA 降至 20 维。
  2. 在真实训练数据上拟合 GaussianMixture，n_components 从 `[1,2,3,5,8]` 中由 BIC 准则自动选取最优值。
  3. `dsi_synth_ll` = GMM 对合成数据（连续列）的对数似然均值。
  4. `dsi_real_ll` = GMM 对真实留出数据的对数似然均值（参考上界）。
  5. `dsi_gap` = `dsi_real_ll − dsi_synth_ll`（绝对差距）。gap 越接近 0，说明合成数据在真实分布下的密度与真实数据相当。
  6. `dsi_relative_gap` = `dsi_gap / |dsi_real_ll| × 100`（相对百分比）。**绝对 gap 依赖数据集维度和量纲，跨数据集无可比性**；相对百分比消除了量纲影响，可直接判断质量好坏：通常 < 5% 为优秀，5–15% 为良好，> 15% 需关注。

- **`icvr`**（需 `LogicSpec.known_fds`）：对每条函数依赖 LHS→RHS，在合成数据中找与真实数据 LHS 值相同的行，检查其 RHS 是否也一致；违背行数 / 可检查总行数即为违背率。

- **`hcs_violation_rate`**（需 `LogicSpec.hierarchies`）：对每条层次约束（如"city 必须属于其对应的 state"），逐行检查合成数据中父子类别的拓扑关系；违背行数 / 总行数即为违背率。

- **`mdi_mean`**（需 `LogicSpec.math_equations`）：对每条数学等式（如 `total_rooms = bedrooms + other_rooms`），计算合成数据的相对误差 `|lhs − rhs| / max(|rhs|, ε)`；超过阈值 ε 的行视为违背；各等式违背率的均值即为 `mdi_mean`。

**DSI 工程细节：**
- 仅对连续列拟合 GMM（避免混合数据类型问题）
- 连续列 > 20 时自动 PCA 降维至 20 维（防维灾）
- BIC 在 `[1,2,3,5,8]` 上自动选最优 n_components
- `dsi_relative_gap` 是计算属性（property），不存储在 JSON 结果文件中，每次从 `dsi_gap` 和 `dsi_real_ll` 动态计算

---

## `dimension3.py` — 下游任务效用评估器

**理论定位：** TSTR（Train-on-Synthetic, Test-on-Real）泛化误差界迁移。若生成分布 Q 完美拟合真实分布 P，则在 Q 上 ERM 训练的模型在 P_test 上的风险应趋近于在 P_train 上训练的基线。效用缺口（utility gap = TRTR − TSTR）量化了生成模型引入的信息损失。

提供两个互补评估模块：

### `MLUtilityEvaluator(schema, target_col, task_type, tuning_mode="shared")`

**核心输出：** `Dim3Result`

| 字段 | 含义 |
|---|---|
| `mle_tstr` | 各模型 TSTR 分数：`{"XGBoost": {"roc_auc": ..., "macro_f1": ...}, ...}` |
| `mle_trtr` | 各模型 TRTR 基线分数（传入 `train_real_df` 时计算） |
| `utility_gap` | TRTR − TSTR，正值表示真实训练数据优于合成数据 |
| `tuning_mode` | 实际使用的调参模式（`"shared"` 或 `"independent"`） |

**评估指标：**

| 任务类型 | 指标 | 方向 |
|---|---|---|
| 分类（二分类 / 多分类） | `roc_auc`（多分类用 OVR 宏平均） | ↑ |
| 分类 | `macro_f1` | ↑ |
| 回归 | `r2` | ↑ |
| 回归 | `wmape`（加权 MAPE，规避零值除法） | ↓ |

**模型矩阵（分类）：** XGBoost · RandomForest · LogisticRegression
**模型矩阵（回归）：** XGBoost · RandomForest · Ridge

**超参调优模式（`tuning_mode`）：**

| 模式 | 调参次数 | 原理 | 适用场景 |
|---|---|---|---|
| `"shared"`（**默认**） | 每模型 1 次（在真实数据上） | 在真实训练集上找最优超参，将**同一套参数**同时用于 TRTR fit 和 TSTR fit。超参不再是混淆变量，utility gap 纯粹反映数据质量差距 | 推荐用于正式实验 |
| `"independent"` | 每模型 2 次（各自独立调参） | TSTR 和 TRTR 分别在各自的训练集上独立搜索最优超参（原始行为） | 需要与旧结果对比时使用 |

优先使用 Optuna TPE（贝叶斯优化），缺少 `optuna` 时自动 fallback 到 `RandomizedSearchCV` 并给出 `ImportWarning`。调优结果记录在 `Dim3Result.tuning_backend` 和 `Dim3Result.tuning_mode` 中。

**通过 `PipelineConfig` 控制：**

```python
PipelineConfig(
    ...
    dim3_tuning_mode="shared",       # 默认，推荐
    # dim3_tuning_mode="independent", # 恢复旧行为
)
```

**各指标计算方式：**

- **`mle_tstr`**：
  1. 特征预处理：ColumnTransformer 在**真实测试集**上 fit（域锚点），避免合成训练集统计污染。连续列 StandardScaler，分类列 OrdinalEncoder（`handle_unknown='use_encoded_value'`）。
  2. 在**合成数据**上训练分类/回归模型（XGBoost / RandomForest / LogisticRegression 或 Ridge）。
  3. 在**真实测试集**上评估：分类任务输出 `roc_auc`（多分类用 OVR 宏平均）和 `macro_f1`；回归任务输出 `r2` 和 `wmape`。
  4. 超参：`"shared"` 模式下在真实训练集上用 Optuna TPE（30 次试验）搜索最优超参，同一套参数用于 TSTR 和 TRTR；`"independent"` 模式下各自独立调参。

- **`mle_trtr`**：同 TSTR 流程，但训练集替换为**真实训练集**，作为性能上限基线。

- **`utility_gap`**：`mle_trtr[metric] − mle_tstr[metric]`，正值表示真实数据优于合成数据的程度。每个模型（XGBoost / RF / LR）各自计算一个 gap 值。

**防御机制：**
- `OrdinalEncoder(handle_unknown='use_encoded_value')` 防止合成/真实数据类别集不一致导致的崩溃
- 预处理在真实测试集上 fit（域锚点），避免合成训练集统计信息污染特征空间
- `"shared"` 模式下若 `train_real_df` 未提供，自动 fallback 并 warning
- 训练集退化（单一类别）时返回空字典而非抛出异常

### `LLMUtilityEvaluator(target_col, task_type)`（可选，需 GPU）

将表格行序列化为 Instruction-Tuning 格式，在合成数据上 LoRA 微调轻量 LLM，在真实测试集上推理并计算分类指标。捕捉树模型无法感知的非线性语义结构。

**依赖：** `pip install transformers peft trl accelerate datasets`
**默认基座：** `Qwen/Qwen2.5-1.5B-Instruct`（可通过 `base_model` 参数覆盖）
**LoRA 配置：** rank=8, alpha=16, dropout=0.05, target_modules="all-linear"

| 字段 | 含义 |
|---|---|
| `lle_tstr` | LLM 在真实测试集上的 `macro_f1` 和 `roc_auc`（二分类） |
| `lle_model` | 使用的基座模型标识符 |

序列化格式：`Input: [col1: val1, col2: val2, ...] Output: <label>`，通过字符串匹配解析预测标签，未匹配时 fallback 到第一个已知类别。

---

## `dimension4.py` — 隐私保护与防记忆评估器

**理论定位：** 双轨制隐私评估。黑盒轨道通过异构 L1 距离（连续特征 MinMax 归一化 + 分类特征 Hamming 0/1）度量合成数据与真实训练集的最近距离，直接量化样本级"复制粘贴"风险；白盒轨道通过掩码条件困惑度（Masked Conditional PPL）从参数层面探测生成器 LLM 是否背诵了训练数据。

### `PrivacyEvaluator(schema, real_train_df)`

**核心输出：** `Dim4Result`（黑盒字段）

| 字段 | 含义 | 方向 |
|---|---|---|
| `dcr_5th_percentile` | DCR 分布第 5 百分位数（最近记录距离下界）| ↑ 越高越安全 |
| `dcr_95th_percentile` | DCR 分布第 95 百分位数（最远记录距离上界）| ↓ 越小越真实 |
| `exact_match_rate` | DCR < ε 的合成行占比（近似复制率）| ↓ 越低越好 |
| `distance_strategy` | 实际使用的计算策略（诊断用）| — |

**混合 L1 距离：**
```
dist(x, y) = Σ |x_cont − y_cont|   (MinMax 归一化至 [0,1])
           + Σ 1(x_cat ≠ y_cat)    (Hamming 0/1 指示函数)
```
等价于 `Σ |x_cont − y_cont| + 0.5 × ||OneHot(x_cat) − OneHot(y_cat)||₁`，与 GReaT / GraDe 论文定义严格对齐。

**策略自动路由：**
- **FAISS IndexFlatL1**：类别特征最大基数 ≤ 200 且 faiss 已安装时启用。分类特征 One-hot 后乘以 0.5 还原 Hamming 等价关系，与连续特征拼接送入 FAISS。最快路径。
- **分块 Numpy 广播（Chunked Broadcasting）**：高基数类别（如邮政编码）或 faiss 未安装时的 fallback。双重分块（默认 chunk_size=512），峰值内存 O(chunk²×n_features)，不构造完整 N×M 矩阵。
- 两条路径实现同一距离度量，结果等价。

**各指标计算方式：**

- **`dcr_5th_percentile`**：
  1. 对每个合成行 s，计算其到所有真实训练行的混合 L1 距离（连续列 MinMax 归一化后取绝对差，分类列取 Hamming 0/1），取其中最小值 `dcr(s)`。
  2. 收集所有合成行的 `dcr` 值，取第 5 百分位数。值越高说明即使是最接近真实数据的那 5% 合成样本也保持了足够距离，隐私保护越好。

- **`dcr_95th_percentile`**：
  1. 同上计算所有合成行的 `dcr(s)`。
  2. 取第 95 百分位数，即最"离群"的那 5% 合成样本的距离上界。
  3. 值越小说明合成数据整体贴近真实分布，模型没有生成无意义的离群噪声；值过大则说明模型在某些区域生成了与真实数据差异极大的样本，保真度差。
  4. **与 `dcr_5th` 配合解读**：前者是隐私安全下界（越大越好），后者是保真度上界（越小越好），两者共同约束"离真实数据不能太近也不能太远"。

- **`exact_match_rate`**：`dcr(s) < ε` 的合成行占总合成行的比例（近似复制率）；ε 默认为归一化距离空间中的极小值（所有列均完全匹配时距离为 0）。

- **`dlt_masked_ppl_train` / `dlt_masked_ppl_test`**（需生成器权重）：
  1. 将真实训练/测试行序列化为 `ColumnA is ValueA, ColumnB is ValueB` 格式。
  2. 用 `return_offsets_mapping` 精确定位值 Token 的字符边界，将列名和分隔符的 label 设为 -100（PyTorch ignore index）。
  3. 在生成器 LLM 上前向传播，仅对值 Token 计算交叉熵损失，取 exp 得到 Masked PPL。

- **`dlt_gap`**：`dlt_masked_ppl_test − dlt_masked_ppl_train`。若生成器严重记忆了训练集，train PPL 会异常低（模型能"背诵"训练值），gap 会显著为正。

**防御机制：** `exact_match_rate > exact_match_warn_rate`（默认 1%）时触发 `DataCopyingWarning`，明确提示隐私风险。

### `MemorizationProbe(model, tokenizer)`（可选，需生成器权重）

将用户的**生成器 LLM**（产生合成数据的模型）传入，通过 Masked Conditional PPL 探测其对训练集的记忆程度。需要访问生成器权重（API 闭源模型无法使用）。

**掩码机制：** 序列化格式 `ColumnA is ValueA, ColumnB is ValueB`，利用 tokenizer 的 `return_offsets_mapping` 精确定位值 Token 的字符边界，将列名和分隔符的 label 设为 -100（PyTorch ignore index），仅对真实数据值 Token 计算交叉熵损失。

| 字段 | 含义 | 方向 |
|---|---|---|
| `dlt_masked_ppl_train` | 生成器在真实训练集上的 Masked PPL | ↑ 越高越安全 |
| `dlt_masked_ppl_test` | 生成器在真实测试集上的 Masked PPL | — |
| `dlt_gap` | PPL(test) − PPL(train)，大正值 = 严重记忆 | → 0 越好 |

**与 Dim3 LLE 的区别：** LLE 是我们自己创建评估用的 LLM（内部工具，测效用）；DLT 是用户把生成器传进来（外部模型，测记忆）。

---

## `dimension5.py` — 公平性与去偏评估器

**理论定位：** 双轨制偏差审计。内在偏差轨道通过归一化互信息（NMI）量化合成数据是否放大或抑制了真实数据中存在的人口属性–预测目标相关性；下游差异轨道通过固定容量 XGBoost 探针（TSTR 协议）在真实测试集上按保护属性分组，评估预测结果的群体间差异。

**设计原则：**
- **FairSpec** 为纯规格对象（无数据），类比 Dim2 的 `LogicSpec`
- KBinsDiscretizer 在 `__init__` 中 fit 于真实训练集（域锚点，NMI 和分组切割共用同一 bin 边界）
- 特征预处理（ColumnTransformer）在 `evaluate()` 中 fit 于真实测试集（遵循 Dim3 惯例，防止合成训练集统计信息泄漏）
- 样本量 < `min_group_size`（默认 30）的分组触发 `GroupCollapseWarning` 并返回 NaN，绝不做 Laplace 平滑（平滑会产生虚假低差异值）
- 内部 XGBoost 探针使用固定超参（n_estimators=100, max_depth=6），与 Dim3 的 Optuna 调优完全解耦（目的是审计偏差，不是最大化预测性能）
- 交叉公平性（Intersectional Fairness）为 opt-in，防止组合爆炸

### `FairSpec`

```python
FairSpec(
    protected_cols=["gender", "race"],
    target_col="income",
    task_type=TaskType.BINARY_CLASS,
    intersectional=False,   # 默认关闭
    min_group_size=30,      # CLT 经验阈值
    n_bins_continuous=5,    # 连续属性等频分箱数
)
```

### `FairnessEvaluator(schema, fair_spec, real_train_df, random_state=42)`

**接口：** `evaluate(synth_df, test_real_df) → Dim5Result`

**核心输出：** `Dim5Result`

#### 轨道 A：内在偏差（NMI）

| 字段 | 含义 | 方向 |
|---|---|---|
| `bias_nmi_real` | 每个保护属性在**真实训练集**上与 target 的 NMI（参考基线）| — |
| `bias_nmi_synth` | 每个保护属性在**合成数据**上的 NMI（审计对象）| ↓ 越低偏差越小 |
| Δ = nmi_synth − nmi_real | 正值 = 放大偏差；负值 = 抑制偏差（可能过度修正）| → 0 |

**各指标计算方式：**

- **`bias_nmi_real` / `bias_nmi_synth`**：
  1. 连续型保护属性先通过 KBinsDiscretizer（等频分箱，n_bins=5，在真实训练集 fit）离散化为类别。
  2. 调用 `sklearn.metrics.normalized_mutual_info_score(protected_col, target_col)` 分别在真实训练集和合成数据上计算 NMI，公式为 `NMI(A;Y) = MI(A;Y) / sqrt(H(A)·H(Y))` ∈ [0,1]。

- **`delta_dp`**（Demographic Parity，人口统计均等）：在真实测试集上，按保护属性分组，计算各组的**正例预测率**（predicted positive rate）；`delta_dp` = 各组正例预测率的最大值 − 最小值。

- **`delta_eop`**（Equal Opportunity，机会均等）：按保护属性分组，计算各组的 **TPR**（True Positive Rate，真正例率）；`delta_eop` = 各组 TPR 的最大值 − 最小值。

- **`delta_eo`**（Equalized Odds，均等化赔率）：同时考虑 TPR 和 FPR（False Positive Rate）；`delta_eo = max(max组间TPR差, max组间FPR差)`。比 ΔEOP 更严格，兼顾受益侧（TPR）和伤害侧（FPR）。

- **`stat_parity_gap`**（回归任务）：按保护属性分组，计算各组的**预测均值**；`stat_parity_gap` = 各组预测均值的最大值 − 最小值。

- **`intersectional_delta_dp`**（需 `intersectional=True`）：将所有保护属性进行笛卡尔积分组（如 "Female×Black"），在组合分组上计算 ΔDP 或 ΔSPG。

> **TSTR 探针配置（Dim5 内部固定）：** XGBoost，n_estimators=100, max_depth=6，不调参——目的是审计偏差而非最大化性能。特征预处理在真实测试集上 fit，与 Dim3 惯例一致。

**NMI 公式：** `NMI(A;Y) = MI(A;Y) / sqrt(H(A)·H(Y))` ∈ [0, 1]，由 `sklearn.metrics.normalized_mutual_info_score` 计算。连续属性先通过共享 KBinsDiscretizer 离散化。

#### 轨道 B：下游差异（TSTR 探针）

**分类任务（BINARY_CLASS / MULTI_CLASS）：**

| 字段 | 定义 | 方向 |
|---|---|---|
| `delta_dp` | ΔDP：max 组间正例预测率差 | ↓ 越小越公平 |
| `delta_eop` | ΔEOP（Equal Opportunity，Hardt 2016）：max 组间 TPR 差 | ↓ |
| `delta_eo` | ΔEO（Equalized Odds，Hardt 2016）：max(max-TPR 差, max-FPR 差) | ↓ |

> **ΔEOP vs ΔEO 的区别：** ΔEOP 只测 TPR（受益侧）；ΔEO 额外测 FPR（伤害侧），例如金融放贷场景中对少数群体的假阳性率偏高同样是歧视。

多分类（MULTI_CLASS）：OvR（One-vs-Rest）策略，所有指标对各类别进行宏平均。

**回归任务（REGRESSION）：**

| 字段 | 定义 | 方向 |
|---|---|---|
| `stat_parity_gap` | ΔSPG：max 组间预测均值差 | ↓ |

**交叉公平性（intersectional=True）：**

| 字段 | 含义 |
|---|---|
| `intersectional_delta_dp` | 所有保护属性笛卡尔积分组（如"Female×Black"）上的 ΔDP（分类）或 ΔSPG（回归） |

**诊断字段：**

| 字段 | 含义 |
|---|---|
| `group_collapse_warnings` | 样本量不足被排除的分组描述列表 |

**防御机制总结：**
- 少于 2 个有效分组时，差异指标全部返回 NaN（差异无意义）
- `np.nanmax / np.nanmin` 聚合确保有效分组的指标不被 NaN 传染
- 特征编码失败 / 探针训练失败 / 探针预测失败 均会提前返回，且携带已计算的 NMI 值（部分结果）

---

## `../visualization/aggregator.py` — 多模型聚合与可视化

**功能定位：** 将 Dim0–Dim5 的单次评估结果跨模型聚合，生成三类出版级图表。**不修改任何评估代码（dimension0-5.py）。**

**核心类：** `ResultAggregator(baseline_model, dcr_reference=None, composite_weights=None)`

| 参数 | 含义 |
|---|---|
| `baseline_model` | 雷达图 `"baseline"` 归一化时的参考模型名称 |
| `dcr_reference` | 真实留存集自 DCR（用于 Privacy 轴绝对归一化）；缺省时退化为 minmax + UserWarning |
| `composite_weights` | 六维加权综合分权重字典，默认 {structural:0.15, fidelity:0.20, logic:0.20, utility:0.20, privacy:0.15, fairness:0.10} |

**核心方法：**

| 方法 | 返回 | 说明 |
|---|---|---|
| `add_model(name, result0-5)` | `None` | 注册一次评估运行；同名多次调用自动累积为多轮运行（用于 Bootstrap CI） |
| `to_leaderboard(n_boot, ci, compact)` | `pd.DataFrame` | 所有原始指标 + 综合分 + Bootstrap CI（≥2 次运行时）；`compact=True` 返回 7 列摘要 |
| `plot_radar(normalize, ...)` | `matplotlib.Figure` | 六轴极坐标雷达图，支持 `"minmax"` 或 `"baseline"` 归一化策略 |
| `plot_pareto(x_metric, y_metric, ...)` | `matplotlib.Figure` | 单一二维 Pareto 前沿图，Pareto 最优点标注 ★ |
| `plot_trade_offs(...)` | `matplotlib.Figure` | 三联 Pareto 图：A(效用×隐私) / B(效用×公平) / C(隐私×公平) |

**雷达轴归一化说明：**

- `Privacy` 轴：若提供 `dcr_reference`，归一化为 `DCR / dcr_reference`；否则为 `DCR / max_DCR`（相对）
- `Logic` 轴：`1 / (1 + dsi_gap)` 映射 [0,∞) → (0,1]，无需外部参考值；面向人类读者的 summary 和排行榜额外展示 `dsi_relative_gap`（相对百分比，跨数据集可比）
- `Fidelity` 轴：内含 `1 - |C2ST_AUC - 0.5| / 0.5`（C2ST 偏离理想值越大，保真度分越低）
- 其他轴：直接使用原始分或 1-metric（越高越好方向统一）

**Bootstrap CI 实现：**
- `n_boot` 次有放回重采样 → 计算综合分均值分布 → 取 `(1-ci)/2` 和 `(1+ci)/2` 分位数
- 单次运行（无法 Bootstrap）：`ci_lo = ci_hi = NaN`

# Experiment 2 — Concept Drift Datasets

## 实验目标

在存在概念漂移（Concept Drift）的流式数据集上评测四种 LLM 驱动的表格合成模型（GReaT、PAFT、GraDe、GraFT），验证各模型在分布随时间变化的数据上的合成质量。

## 数据集

| 数据集 | 特征数 | 样本数（train/test） | 漂移类型 | 说明 |
|---|---|---|---|---|
| agrawal | 10 | 16000 / 4000 | 突变漂移 | 贷款审批场景，特征间规则随时间切换 |
| hyperplane | 10 | 16000 / 4000 | 渐变漂移 | 高维超平面旋转，分类边界持续移动 |
| rbfdrift | 10 | 16000 / 4000 | 渐变漂移 | 基于 RBF 的高斯分布中心漂移 |
| sea | 3 | 16000 / 4000 | 突变漂移 | 简单阶梯函数，决策阈值周期切换 |
| stagger | 3 | 16000 / 4000 | 突变漂移 | 全类别特征，规则集离散切换 |

所有数据集均为**二分类任务**，目标列为 `y`。

列 `t`（时间步序号，0~15999）在训练前统一丢弃，不作为特征输入。

## 实验设计

- **训练数据**：`origin_data/<dataset>/train.csv`（含完整漂移过程）
- **测试数据**：`origin_data/<dataset>/test.csv`（漂移后分布）
- **合成输出**：`synthetic_data/<ModelName>/<dataset>_synth.csv`

## 评测维度

| 维度 | 内容 |
|---|---|
| Dim0 | 结构有效性（IRR 无效行率） |
| Dim1 | 分布保真度（KS / TVD / α-precision / β-recall / C2ST） |
| Dim2 | 逻辑一致性（DSI；无 LogicSpec，ICVR/HCS/MDI 跳过） |
| Dim3 | ML 效用（TSTR，二分类，target=`y`） |
| Dim4 | 隐私性（DCR 第5/95百分位数） |
| Dim5 | 跳过（无保护属性） |

## 运行方法

```bash
# 第一步：生成合成数据（当前仅激活 agrawal）
python generate_synthetic2.py

# 激活更多数据集：编辑 generate_synthetic2.py 中的 DATASET_CONFIGS，
# 取消对应数据集的注释，同时在 run_benchmark2.py 的 DATASETS 中同步取消注释。

# 第二步：评测
python run_benchmark2.py
```

## 结果输出

- `results2/<model>_<dataset>.json` — 每次运行的完整 ResultBundle
- `results2/leaderboard.csv` — 汇总排行榜
- `experiment2_log.csv` / `experiment2_log.jsonl` — 训练与采样耗时记录

## 与实验1的区别

| | 实验1 | 实验2 |
|---|---|---|
| 数据类型 | 静态表格（无漂移） | 流式数据（含概念漂移） |
| 数据集 | diabetes / house / income / sick / us_location | agrawal / hyperplane / rbfdrift / sea / stagger |
| 目标列 | 各数据集不同 | 统一为 `y`（二分类） |
| 特殊处理 | 无 | 丢弃时间列 `t` |
| Dim5 公平性 | income 数据集开启 | 全部跳过 |

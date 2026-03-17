import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_causal_data(n_samples, scenario="normal"):
    """
    这是一个简单的SCM执行引擎（采样层）
    scenario: "normal" (正常) 或 "recession" (LLM干预后的经济危机)
    """
    # 1. 外生变量 (Exogenous Noise) - 模拟个体差异
    U_income = np.random.normal(0, 1, n_samples)
    U_score  = np.random.normal(0, 1, n_samples)
    U_default = np.random.uniform(0, 1, n_samples)

    # 2. 根节点：年收入 (Income) - 假设正常分布在 50k-150k
    Income = 100 + 25 * U_income 
    
    # 3. 信用评分 (Credit Score) - 受收入影响
    # 方程：Score = 0.5 * Income + Noise
    Credit_Score = 0.6 * Income + 10 * U_score + 300

    # --- 核心区别：LLM 修改后的结构方程 (Structural Equations) ---
    
    if scenario == "normal":
        # 正常逻辑：贷款额度 = 收入的 4 倍 + 信用评分修正
        Loan_Amount = (Income * 4) + (Credit_Score * 0.1) + np.random.normal(0, 5, n_samples)
        
        # 正常逻辑：违约概率低，主要受 (贷款/收入比) 影响
        # 逻辑：如果贷款远超收入，则容易违约
        default_logic = (Loan_Amount / Income) * 0.1 + (U_default * 0.2)
        Default = (default_logic > 0.6).astype(int)

    elif scenario == "recession":
        # LLM 干预：银行变谨慎了，贷款额度减半
        # 方程改变：系数从 4 变成 2
        Loan_Amount = (Income * 2) + (Credit_Score * 0.05) + np.random.normal(0, 2, n_samples)
        
        # LLM 干预：经济危机下，外生环境导致违约率整体上升 30%
        # 方程改变：加入了基础违约常数 0.4
        default_logic = (Loan_Amount / Income) * 0.1 + 0.4 + (U_default * 0.3)
        Default = (default_logic > 0.7).astype(int)

    # 封装数据
    df = pd.DataFrame({
        'Income': Income,
        'Credit_Score': Credit_Score,
        'Loan_Amount': Loan_Amount,
        'Default': Default
    })
    return df

# --- 实验演示 ---

# 1. 生成数据
n = 1000
df_normal = generate_causal_data(n, scenario="normal")
df_recession = generate_causal_data(n, scenario="recession")

# 2. 观察逻辑变化（可视化）
plt.figure(figsize=(12, 5))

# 观察收入与贷款额度的关系变化
plt.subplot(1, 2, 1)
sns.regplot(data=df_normal, x='Income', y='Loan_Amount', label='Normal', scatter_kws={'alpha':0.3})
sns.regplot(data=df_recession, x='Income', y='Loan_Amount', label='Recession (LLM Adjusted)', color='red', scatter_kws={'alpha':0.3})
plt.title("LLM Intervention: Income vs Loan Amount")
plt.legend()

# 观察违约率的变化 (OOD 表现)
plt.subplot(1, 2, 2)
combined_df = pd.concat([
    df_normal.assign(Period='Normal'),
    df_recession.assign(Period='Recession')
])
sns.barplot(data=combined_df, x='Period', y='Default')
plt.title("Systemic Risk: Default Rate Increase")

plt.tight_layout()
plt.show()

print("正常时期平均违约率:", df_normal['Default'].mean())
print("危机时期平均违约率:", df_recession['Default'].mean())
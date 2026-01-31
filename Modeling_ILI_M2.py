import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Color scheme (consistent across all plots)
# ============================================================
virus_colors = {
    "Flu_A": "#1f77b4",        # blue
    "Flu_B": "#ff7f0e",        # orange
    "RSV": "#2ca02c",          # green
    "Virus_Total": "#d62728",  # red
    "ILI": "#9e9e9e"           # gray
}

virus_palette = {
    "A_rate": virus_colors["Flu_A"],
    "B_rate": virus_colors["Flu_B"],
    "RSV": virus_colors["RSV"]
}

# ============================================================
# 1. Load and preprocess data
# ============================================================
file_path = "your/local/path/flu_rsv.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")
df.columns = df.columns.str.strip()

df = df[df['ISO_YEAR'].between(2022, 2023)]
df['ISO_WEEKSTARTDATE'] = pd.to_datetime(df['ISO_WEEKSTARTDATE'])

cols = ['A_rate', 'B_rate', 'RSV_rate', 'ILI', 'Population']
df[cols] = df[cols].fillna(0)

df = df.sort_values('ISO_WEEKSTARTDATE').reset_index(drop=True)
df['t'] = np.arange(len(df))

# ============================================================
# 2. Predictors + extended seasonality (remove 8-week cycle)
# ============================================================
epsilon = 1e-6
df['A_rate10'] = df['A_rate'] * 10 + epsilon
df['B_rate10'] = df['B_rate'] * 10 + epsilon
df['RSV_rate10'] = df['RSV_rate'] * 10 + epsilon

# Only 52, 26, 13-week cycles
for period in [52, 26, 13]:
    df[f'cos_{period}'] = np.cos(2 * np.pi * df['t'] / period)
    df[f'sin_{period}'] = np.sin(2 * np.pi * df['t'] / period)

X_cols = ['A_rate10','B_rate10','RSV_rate10'] + \
         [f'cos_{p}' for p in [52,26,13]] + \
         [f'sin_{p}' for p in [52,26,13]]
X = sm.add_constant(df[X_cols])

y = df['ILI'] / 1000
offset = np.log(df['Population'] / 1000 + 1)

# ============================================================
# 3. Negative Binomial GLM
# ============================================================
model = sm.GLM(
    y,
    X,
    family=sm.families.NegativeBinomial(alpha=0.5),
    offset=offset
)

result = model.fit()
print(result.summary())

# ============================================================
# 4. Virus-attribution function
# ============================================================
def virus_attribution(beta):
    mu_full = np.exp(X @ beta + offset) * 1000
    X_noA = X.copy(); X_noA['A_rate10'] = epsilon
    X_noB = X.copy(); X_noB['B_rate10'] = epsilon
    X_noR = X.copy(); X_noR['RSV_rate10'] = epsilon
    mu_noA = np.exp(X_noA @ beta + offset) * 1000
    mu_noB = np.exp(X_noB @ beta + offset) * 1000
    mu_noR = np.exp(X_noR @ beta + offset) * 1000
    return pd.DataFrame({
        'Flu_A': (mu_full - mu_noA).clip(lower=0),
        'Flu_B': (mu_full - mu_noB).clip(lower=0),
        'RSV':   (mu_full - mu_noR).clip(lower=0)
    })

# ============================================================
# 5. Simulation-based 95% CI
# ============================================================
np.random.seed(42)
n_sim = 5000
coef_sim = np.random.multivariate_normal(result.params, result.cov_params(), size=n_sim)

sim_results = []
for b in coef_sim:
    sim = virus_attribution(b)
    yearly = sim.groupby(df['ISO_YEAR']).sum()
    yearly['Virus_Total'] = yearly.sum(axis=1)
    sim_results.append(yearly)

sim_df = pd.concat(sim_results, keys=range(n_sim), names=['sim','ISO_YEAR'])
ci_lower = sim_df.groupby('ISO_YEAR').quantile(0.025)
ci_upper = sim_df.groupby('ISO_YEAR').quantile(0.975)

# ============================================================
# 6. Point estimates
# ============================================================
mu_full = np.exp(X @ result.params + offset) * 1000
df['Predicted_ILI'] = mu_full

df['Flu_A'] = (mu_full - np.exp(X.assign(A_rate10=epsilon) @ result.params + offset) * 1000).clip(lower=0)
df['Flu_B'] = (mu_full - np.exp(X.assign(B_rate10=epsilon) @ result.params + offset) * 1000).clip(lower=0)
df['RSV']   = (mu_full - np.exp(X.assign(RSV_rate10=epsilon) @ result.params + offset) * 1000).clip(lower=0)
df['Virus_Total'] = df[['Flu_A','Flu_B','RSV']].sum(axis=1)

# ============================================================
# 7. Combined plot: A) Model Fit and B) Virus-attributable ILI
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14,10), sharex=True)

# A) Model Fit
axes[0].scatter(df['ISO_WEEKSTARTDATE'], df['ILI'],
                color=virus_colors["ILI"], alpha=0.7, s=25,
                label='Observed ILI')
axes[0].plot(df['ISO_WEEKSTARTDATE'], df['Predicted_ILI'],
             color='black', linestyle='--', linewidth=2,
             label='Predicted ILI (Model)')
axes[0].set_ylabel("ILI count")
axes[0].legend()
axes[0].set_title("A) Model Fit")

# B) Virus-attributable ILI
axes[1].bar(df['ISO_WEEKSTARTDATE'], df['ILI'],
            color=virus_colors["ILI"], alpha=0.4,
            label='Observed ILI')
axes[1].plot(df['ISO_WEEKSTARTDATE'], df['Flu_A'],
             color=virus_colors["Flu_A"], label='Influenza A')
axes[1].plot(df['ISO_WEEKSTARTDATE'], df['Flu_B'],
             color=virus_colors["Flu_B"], label='Influenza B')
axes[1].plot(df['ISO_WEEKSTARTDATE'], df['RSV'],
             color=virus_colors["RSV"], label='RSV')
axes[1].plot(df['ISO_WEEKSTARTDATE'], df['Virus_Total'],
             color=virus_colors["Virus_Total"], linestyle='--',
             label='Total Virus')
axes[1].set_xlabel("Week")
axes[1].set_ylabel("ILI count")
axes[1].legend()
axes[1].set_title("B) Virus-attributable ILI")

plt.tight_layout()
plt.show()

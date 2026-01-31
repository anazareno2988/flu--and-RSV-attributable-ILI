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
    "RSV_rate": virus_colors["RSV"]
}

# ============================================================
# 1. Load and preprocess data
# ============================================================
file_path = "/Users/allennazareno/Documents/flu_rsv.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")
df.columns = df.columns.str.strip()

df = df[df['ISO_YEAR'].between(2022, 2023)]
df['ISO_WEEKSTARTDATE'] = pd.to_datetime(df['ISO_WEEKSTARTDATE'])

cols = ['A_rate', 'B_rate', 'RSV_rate', 'ILI', 'Population']
df[cols] = df[cols].fillna(0)

df = df.sort_values('ISO_WEEKSTARTDATE').reset_index(drop=True)
df['t'] = np.arange(len(df))

# ============================================================
# 2. Predictors + seasonality
# ============================================================
epsilon = 1e-6

df['A_rate10'] = df['A_rate'] * 10 + epsilon
df['B_rate10'] = df['B_rate'] * 10 + epsilon
df['RSV_rate10'] = df['RSV_rate'] * 10 + epsilon

df['cos_52'] = np.cos(2 * np.pi * df['t'] / 52)
df['sin_52'] = np.sin(2 * np.pi * df['t'] / 52)

X = df[['A_rate10', 'B_rate10', 'RSV_rate10', 'cos_52', 'sin_52']]
X = sm.add_constant(X)

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
# 4. Attribution function
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

coef_sim = np.random.multivariate_normal(
    result.params,
    result.cov_params(),
    size=n_sim
)

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

df['Flu_A'] = (mu_full - np.exp(X.assign(A_rate10=epsilon) @ result.params + offset) * 1000).clip(lower=0)
df['Flu_B'] = (mu_full - np.exp(X.assign(B_rate10=epsilon) @ result.params + offset) * 1000).clip(lower=0)
df['RSV']   = (mu_full - np.exp(X.assign(RSV_rate10=epsilon) @ result.params + offset) * 1000).clip(lower=0)

df['Virus_Total'] = df[['Flu_A','Flu_B','RSV']].sum(axis=1)

# ============================================================
# 7. Time series plot 
# ============================================================
plt.figure(figsize=(14,6))

plt.bar(df['ISO_WEEKSTARTDATE'], df['ILI'],
        color=virus_colors["ILI"], alpha=0.4, label='Observed ILI')

plt.plot(df['ISO_WEEKSTARTDATE'], df['Flu_A'],
         color=virus_colors["Flu_A"], label='Influenza A')

plt.plot(df['ISO_WEEKSTARTDATE'], df['Flu_B'],
         color=virus_colors["Flu_B"], label='Influenza B')

plt.plot(df['ISO_WEEKSTARTDATE'], df['RSV'],
         color=virus_colors["RSV"], label='RSV')

plt.plot(df['ISO_WEEKSTARTDATE'], df['Virus_Total'],
         color=virus_colors["Virus_Total"], linestyle='--', label='Total Virus')

plt.xlabel("Week")
plt.ylabel("ILI count")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 8. Boxplots of ILI and Virus Rates 
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12,6), sharey=False)

# Panel 1: ILI (gray)
sns.boxplot(
    data=df,
    x='ISO_YEAR',
    y='ILI',
    ax=axes[0],
    palette=[virus_colors["ILI"], virus_colors["ILI"]]
)
axes[0].set_xlabel("Year")
axes[0].set_ylabel("ILI count")

# Panel 2: Virus rates
virus_df = df.melt(
    id_vars='ISO_YEAR',
    value_vars=['A_rate', 'B_rate', 'RSV_rate'],
    var_name='Virus',
    value_name='Rate'
)

sns.boxplot(
    data=virus_df,
    x='ISO_YEAR',
    y='Rate',
    hue='Virus',
    ax=axes[1],
    palette=virus_palette
)

axes[1].set_xlabel("Year")
axes[1].set_ylabel("Virus rate")

# Rename legend labels
handles, labels = axes[1].get_legend_handles_labels()
new_labels = ["Influenza A", "Influenza B", "RSV"]
axes[1].legend(handles, new_labels, title="Virus")

plt.tight_layout()
plt.show()

# ============================================================
# 9. Predicted ILI from the model
# ============================================================
df['Predicted_ILI'] = (np.exp(X @ result.params + offset) * 1000)

# ============================================================
# 10. Combined plot: A) Model Fit and B) Virus-attributable ILI
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(14,10), sharex=True)

# -----------------------------
# A) Model Fit: Observed vs Predicted
# -----------------------------
axes[0].scatter(df['ISO_WEEKSTARTDATE'], df['ILI'],
                color=virus_colors["ILI"], alpha=0.7, s=25,
                label='Observed ILI')

axes[0].plot(df['ISO_WEEKSTARTDATE'], df['Predicted_ILI'],
             color='black', linestyle='--', linewidth=2,
             label='Predicted ILI (Model)')

axes[0].set_ylabel("ILI count")
axes[0].legend()
axes[0].set_title("A) Model Fit")

# -----------------------------
# B) Virus-attributable ILI (with observed bars)
# -----------------------------
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

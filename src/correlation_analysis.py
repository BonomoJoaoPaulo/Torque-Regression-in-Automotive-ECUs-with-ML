import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASETS_PATH = os.path.join(ROOT_PATH, '../datasets')
PLOT_PATH = os.path.join(ROOT_PATH, '../plots')

# Define target and dataset
target = 'target_torque'
dataset = 'all_data.parquet'

# Load data
df = pd.read_parquet(os.path.join(DATASETS_PATH, dataset))

# Separate features and target
X = df.drop(target, axis=1)
y = df[target]

# Feature Mapping
feature_names = X.columns.tolist()
feature_mapping = {original_name: f"Feature {i+1}" for i, original_name in enumerate(feature_names)}
X_renamed = X.rename(columns=feature_mapping)

# Calculate Pearson and Spearman correlations
pearson_corr = X_renamed.corrwith(y, method='pearson').sort_values(ascending=False)
spearman_corr = X_renamed.corrwith(y, method='spearman').sort_values(ascending=False)

# Create a DataFrame to visualize correlations
corr_df = pd.DataFrame({
    'Pearson': pearson_corr,
    'Spearman': spearman_corr
}).sort_values(by='Pearson', ascending=False)

# Print Correlation DataFrame
print(corr_df)

# Plot top 8 Pearson correlations
plt.figure(figsize=(12, 6))
top8_pearson = corr_df['Pearson'].abs().head(7)
sns.barplot(x=top8_pearson.values, y=top8_pearson.index, palette='Blues_d')
plt.title('Pearson Correlations Between Selected Features and Target (Torque)')
plt.savefig(os.path.join(PLOT_PATH, 'top7_pearson_correlation.png'), dpi=300)
plt.show()

# Plot top 8 Spearman correlations
plt.figure(figsize=(12, 6))
top8_spearman = corr_df['Spearman'].abs().head(8)
sns.barplot(x=top8_spearman.values, y=top8_spearman.index, palette='Greens_d')
plt.title('Spearman Correlations Between Selected Features and Target (Torque)')
plt.savefig(os.path.join(PLOT_PATH, 'top7_spearman_correlation.png'), dpi=300)
plt.show()

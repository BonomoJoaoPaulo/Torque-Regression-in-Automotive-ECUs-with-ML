import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib

# Paths
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(ROOT_PATH, '../datasets')
OUTPUT_DIR = os.path.join(ROOT_PATH, '../outputs')
PLOT_DIR = os.path.join(ROOT_PATH, '../plots')
MODEL_DIR = os.path.join(ROOT_PATH, '../models')

# Load data
df = pd.read_parquet(os.path.join(DATASETS_DIR, 'all_data.parquet'))

# Separate features and target
X = df.drop('target_torque', axis=1)
y = df['target_torque']

# Feature Mapping to not display the original feature confidential names
feature_names = X.columns.tolist()
feature_mapping = {original_name: f"Feature {i+1}" for i, original_name in enumerate(feature_names)}
X_renamed = X.rename(columns=feature_mapping)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_renamed, y, test_size=0.2, shuffle=False)

# Feature selection
k_features = 29
skb = SelectKBest(score_func=f_regression, k=k_features)
skb.fit(X_train, y_train)
X_train_selected = skb.transform(X_train)
X_test_selected = skb.transform(X_test)
selected_features = X_renamed.columns[skb.get_support()]

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Train Ridge Regression model
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train)

# Make predictions
predictions_ridge = ridge_model.predict(X_test_scaled)

# Evaluate the model
rmse_ridge = np.sqrt(mean_squared_error(y_test, predictions_ridge))
mae_ridge = mean_absolute_error(y_test, predictions_ridge)
r2_ridge = r2_score(y_test, predictions_ridge)
print(f"RMSE: {rmse_ridge}, MAE: {mae_ridge}, R²: {r2_ridge}")

# Plot actual vs predicted
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, predictions_ridge, label='Predicted', linestyle='--', color='red')
plt.title('Actual vs Predicted Torque using Ridge Regression')
plt.xlabel('Index')
plt.ylabel('Torque (N.m)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'comparison_over_time_ridge.png'), dpi=300)
plt.show()

# Feature coefficients
coefficients = ridge_model.coef_ 
feature_coefficients = pd.Series(coefficients, index=selected_features).sort_values(key=lambda x: x.abs(), ascending=False)

# Plot feature coefficients
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_coefficients.values, y=feature_coefficients.index, palette='coolwarm')
plt.title('Feature Coefficients in Ridge Regression')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'feature_coefficients_ridge.png'), dpi=300)
plt.show()

# Save the model
joblib_file = os.path.join(MODEL_DIR, 'ridge_regression_model.joblib')
joblib.dump(ridge_model, joblib_file)

print(f"Model saved to {joblib_file}")

import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
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

# Train XGBRegressor model
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train_selected, y_train)

# Make predictions
predictions_xgb = xgb_model.predict(X_test_selected)

# Evaluate the model
rmse_xgb = np.sqrt(mean_squared_error(y_test, predictions_xgb))
mae_xgb = mean_absolute_error(y_test, predictions_xgb)
r2_xgb = r2_score(y_test, predictions_xgb)
print(f"RMSE: {rmse_xgb}, MAE: {mae_xgb}, R²: {r2_xgb}")

# Plot actual vs predicted
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, predictions_xgb, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Torque')
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, 'comparison_over_time.png'), dpi=300)
plt.show()

# Feature importance
importances = xgb_model.feature_importances_
feature_importances = pd.Series(importances, index=selected_features).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='magma')
plt.title('Feature Importance according to XGBoost')
plt.savefig(os.path.join(PLOT_DIR, 'feature_importance_xgboost.png'), dpi=300)
plt.show()

# Save the model
joblib_file = os.path.join(MODEL_DIR, 'xgboost_model.joblib')
joblib.dump(xgb_model, joblib_file)

print(f"Model saved to {joblib_file}")

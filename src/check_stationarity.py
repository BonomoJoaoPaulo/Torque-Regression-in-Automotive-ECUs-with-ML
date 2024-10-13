"""
Originaly, the code was written in a Jupyter Notebook.
"""

from pandas import read_parquet
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
from pathlib import Path

column_name = 'target_torque'

# Datasets filenames ommited
filenames = []

skipfooter = 2


data_folder = 'datasets'
script_directory = Path().resolve()  # jupyter notebook
data_path = script_directory / data_folder

# Define the number of subplots per row and column
subplots_per_row = 3
subplots_per_col = 4
num_subplots = len(filenames)

# Calculate the number of rows and columns
num_rows = num_subplots // subplots_per_row
num_cols = min(subplots_per_row, num_subplots) if num_subplots % subplots_per_row == 0 else subplots_per_col

# Create the figure and axes
f, axes = plt.subplots(num_rows, num_cols, figsize=(21, 15))
f.tight_layout()

# Loop through the filenames and plot the time series
for i, filename in enumerate(filenames):
    row = i // subplots_per_col
    col = i % subplots_per_col
    
    file_path = os.path.join(data_path, filename)

    series = read_parquet(file_path, engine='pyarrow', columns=[column_name])
    
    # Removing the last rows
    series = series.iloc[:-skipfooter]
    
    # Converting the series to a two-dimensional array
    data = series.values.astype('float32')
    
    # Applying MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Ploting the time series
    if num_rows > 1:
        axes[row, col].plot(scaled_data)
        axes[row, col].set_title(filename)
    else:
        axes[i].plot(scaled_data)
        axes[i].set_title(filename)

plt.savefig('../plots/all_scaled_timeseries_comparison.png', facecolor='white')

from pandas import Series
from statsmodels.tsa.stattools import adfuller

for i, filename in enumerate(filenames):

    print(i)
    file_path = os.path.join(data_path, filename)

    series = read_parquet(file_path, engine='pyarrow', columns=[column_name])
    
    # Removing the last rows of the series
    series = series.iloc[:-skipfooter]
    
    # Converting the series to a two-dimensional array
    data = series.values.astype('float32')
    
    P_result = adfuller(data.flatten())

    print(f" {filename}")
    print(f"ADF Statistic: {P_result[0]}")
    print(f"p-value: {P_result[1]}")
    print('Critical Values:')

    names = []
    for name, number in P_result[4].items():
        names.append(name)
        
    for i in range(len(names)):
        value = P_result[4][names[i]]
        print('            {0:2s}:          {1:.4f}'.format(names[i], value))

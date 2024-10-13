import pandas as pd
import numpy as np
import os
import json

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASETS_PATH = os.path.join(ROOT_PATH, 'datasets')
OUTPUT_PATH = os.path.join(ROOT_PATH, 'outputs')
PLOTS_PATH = os.path.join(ROOT_PATH, 'plots')

datasets_list = []

i = 0
for dataset in os.listdir(DATASETS_PATH):
    if dataset.endswith('.parquet'):
        df = pd.read_parquet(os.path.join(DATASETS_PATH, dataset))
        if i != 0:
            df.index = df.index - df.index.min() + datasets_list[i-1].index.max() + pd.Timedelta(seconds=1)
        else:
            pass
        
        datasets_list.append(df)
        i += 1


common_columns = list(set.intersection(*[set(df.columns) for df in datasets_list]))
print("Common columns:\n", common_columns)

final_df = pd.concat([df[common_columns] for df in datasets_list], ignore_index=False)

#final_df.drop(columns=['Feature_36'], inplace=True)

# This function is used to drop the other torque related variables from the dataset,
# such as "Engine torque losses". The only torque variable that should be kept is the target_torque.
def drop_torque_vars(variables_data, df):
    torque_vars = []

    for var in variables_data:
        if "torque" in var['varIdentifier'].lower() and var['name'] != 'target_torque':
            torque_vars.append(var['name'])

    print("Torque variables:\n", torque_vars)
    df.drop(columns=torque_vars, inplace=True)

    return df

variables_data = json.load(open(os.path.join(OUTPUT_PATH, 'all_data_variables_info.json'), 'r'))
drop_torque_vars(variables_data, final_df)

final_df.to_parquet(os.path.join(DATASETS_PATH, "all_data.parquet"))

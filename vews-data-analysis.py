import os

import pandas as pd

def timestamp_to_millis(row, column_name):
    if row[column_name] == '-':
        return -1
    timestamp = pd.Timestamp(row[column_name])
    if type(timestamp) == pd.tslib.Timestamp:
        timestamp = pd.Timestamp(row[column_name])
        return timestamp.value
    else:
        return -1

directory = os.path.join("vews_dataset_v1.1/")

benign_df = pd.DataFrame()
vandal_df = pd.DataFrame()

for root, dirs, files in os.walk(directory):
    for file_name in files:
        if file_name.endswith(".csv"):
            df = pd.read_csv(root + file_name, sep=',', usecols=['username',
                                                                 'revid',
                                                                 'revtime',
                                                                 'pagetitle',
                                                                 'isReverted',
                                                                 'revertTime'])

            df['revtime'] = df.apply(lambda row: timestamp_to_millis(row, 'revtime'), axis=1)
            df['revertTime'] = df.apply(lambda row: timestamp_to_millis(row, 'revertTime'), axis=1)
            if file_name.startswith("benign"):
                df['vandal'] = 0;
                benign_df = benign_df.append(df, ignore_index=True)
            elif file_name.startswith('vandal'):
                df['vandal'] = 1;
                vandal_df = vandal_df.append(df, ignore_index=True)

# Combine benign and vandal dataframes
# print "Combining benign and vandal datasets"
# total_df = benign_df.append(vandal_df, ignore_index=True)

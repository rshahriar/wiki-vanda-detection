import math
import os
import time

import pandas as pd
from sklearn.utils import shuffle


def timestamp_to_millis(row, column_name):
    if row[column_name] == '-':
        return -1
    timestamp = pd.Timestamp(row[column_name])
    if type(timestamp) == pd.tslib.Timestamp:
        timestamp = pd.Timestamp(row[column_name])
        return timestamp.value
    else:
        return -1


def check_metapage(pagetitle):
    if pagetitle.startswith('user:') \
            or pagetitle.startswith('user talk:') \
            or pagetitle.startswith('talk:'):
        return 1
    else:
        return 0


def detect_vandal_by_fm(row):
    vandal = False
    # no vandal detection on meta pages
    if check_metapage(row['pagetitle'].lower()):
        return vandal

    temp = df.loc[(df['username'] == row['username']) & (df['revtime'] <= row['revtime'])]
    temp.sort_values(['username', 'revtime'], ascending=True)

    for i, r in temp.iterrows():
        if check_metapage(r['pagetitle'].lower()):
            vandal = False
        else:
            vandal = True
        return vandal


def detect_vandal_by_crm(row):
    if check_metapage(row['pagetitle'].lower()):
        return False, False, False

    temp = df.loc[(df['username'] == row['username']) & (df['revtime'] <= row['revtime'])]
    temp.sort_values(['username', 'revtime'], ascending=True)

    if len(temp) > 2:
        last_page = temp.iloc[[temp.shape[0]-2]]
        second_last_page = temp.iloc[[temp.shape[0]-2]]

        if check_metapage(last_page.iloc[0]['pagetitle'].lower()) \
            & check_metapage(second_last_page.iloc[0]['pagetitle'].lower())\
            and last_page.iloc[0]['pagetitle'] == second_last_page.iloc[0]['pagetitle']:
                return False, False, False

        else:
            return True, True, True

                # ideal_interval_millis = 180000
                # time_diff = second_last_page.iloc[0]['revtime'] - last_page.iloc[0]['revtime']
                # if time_diff < ideal_interval_millis:
                #     return False, False
                # else:
                #     return True, True

    return False, False, False

def detect_vandal_ntus(row):
    vandal = 1
    # no vandal detection on meta pages
    if check_metapage(row):
        vandal = -1
        return vandal
    # 15 minutes interval
    ideal_interval_millis = 900000
    pagetitle = row['pagetitle']
    temp = df.loc[(df['username'] == row['username']) &
                  (df['revtime'] <= row['revtime'])]
    temp.sort_values(['pagetitle', 'revtime'], ascending=True)
    # benign user if -
    # edit of a new page at a distance of at most 3 hops
    # the gap between two edits are 15 minutes
    rt_list = []
    for i, r in temp.iterrows():
        if i > 3:
            break
        if r['pagetitle'] != pagetitle:
            vandal = -1
        rt_list.append(r['revtime'])
    hop_time_diff = sum([j - i for i, j in zip(rt_list[:-1], rt_list[1:])]) / 2
    if hop_time_diff >= ideal_interval_millis and vandal == 1:
        vandal = -1
    return vandal

# load data from datasets
start = time.time()
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

            # workaround for smaller datasets
            # data_size = df.shape[0]
            # split_point = int(math.ceil(data_size * 0.01))
            # df = df[:split_point]

            # df['isMetapage'] = df.apply(lambda row: check_metapage(row), axis=1)
            df['revtime'] = df.apply(lambda row: timestamp_to_millis(row, 'revtime'), axis=1)
            df['revertTime'] = df.apply(lambda row: timestamp_to_millis(row, 'revertTime'), axis=1)
            if file_name.startswith("benign"):
                df['vandal'] = 0;
                benign_df = benign_df.append(df, ignore_index=True)
            elif file_name.startswith('vandal'):
                df['vandal'] = 1;
                vandal_df = vandal_df.append(df, ignore_index=True)

# Combine benign and vandal dataframes
print "Combining benign and vandal datasets"
total_df = benign_df.append(vandal_df, ignore_index=True)

# sort data by nested sorting on pagetitle and revision time
print "Sorting dataset by pagetitle, username, and revtime"
total_df.sort_values(['pagetitle', 'username', 'revtime'], ascending=True)

# calculate feature vectors
print "adding feature: ntus..."
total_df['ntus'] = total_df.apply(lambda row: detect_vandal_ntus(row), axis=1)
total_df['fm'] = total_df.apply(lambda row: detect_vandal_by_fm(row), axis=1)
total_df[['crmv', 'crmf', 'crms']] = total_df.apply(lambda row: pd.Series(detect_vandal_by_crm(row)), axis=1)

# Remove all metapages - not required anymore
print "Removing all metapages"
total_df = total_df.drop(total_df[total_df.pagetitle.str.lower().startswith('user') |
                                  total_df.pagetitle.str.lower().startswith('user talk') |
                                  total_df.pagetitle.str.lower().startswith('talk')].index)
# to_drop = total_df.applymap(lambda row: check_metapage(row)).all()
# total_df.drop(to_drop)
print "Size of total datasets after removing metapages: ", total_df.shape

# convert True/False to 1/0 respectively
# total_df['isReverted'] = df['isReverted'].astype(int)

# shuffle (to randomize) and split data into training and testing datasets (80%-20%)
total_df = shuffle(total_df)
total_data_size = total_df.shape[0]
split_point = int(math.ceil(total_data_size * 0.80))
train_df, test_df = df[:split_point], df[split_point:]
print "Size of train data set: ", train_df.shape
print "Size of test data set: ", test_df.shape

# write to out folder as csv files
print "Writing dataframes to files..."
train_df.to_csv('out/train.csv', sep=',', encoding='utf-8', index=False)
test_df.to_csv('out/test.csv', sep=',', encoding='utf-8', index=False)

end = time.time()
print "Time to preprocess data: ", math.ceil((end - start)/60), " minutes"

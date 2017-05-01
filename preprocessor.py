import math
import os
import time

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


def check_metapage(row):
    if row['pagetitle'].startswith('User') \
            or row['pagetitle'].startswith('User Talk') \
            or row['pagetitle'].startswith('Talk'):
        return 1
    else:
        return 0


def detect_vandal_by_fm(row, df):
    vandal = False
    # no vandal detection on meta pages
    if check_metapage(row):
        return vandal
    # get the username
    username = row['username']
    revtime = row['revtime']
    pagetitle = row['pagetitle']
    # look up the user records for that page title
    temp = df.loc[df['username'] == username and df['revtime'] <= revtime]
    for i in range(temp.shape[0]):
        # look down in the data frame for the meta page
        # until edit time stamp is reached
        search_page_title = temp.ix[i]['pagetitle']
        if search_page_title.startswith('User') \
                or search_page_title.startswith('User Talk') \
                or search_page_title.startswith('Talk'):
            # if anytime in the past the user edited the metapage,
            # then the user is not a vandal
            if search_page_title == pagetitle:
                vandal = True
    return vandal


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
train_data_frame = pd.DataFrame()
test_data_frame = pd.DataFrame()
for root, dirs, files in os.walk(directory):
    for file_name in files:
        if file_name.endswith(".csv"):
            df = pd.read_csv(root + file_name, sep=',', usecols=['username',
                                                                 'revid',
                                                                 'revtime',
                                                                 'pagetitle',
                                                                 'isReverted',
                                                                 'revertTime'])
            # reducing rows of datasets to have a smaller
            # dataset for fit-predict result - whole
            # UMDWikiDataset is too lengthy for classifier
            # data_size = df.shape[0]
            # split_point = int(math.ceil(data_size * 0.01))
            # df = df[:split_point]

            ####
            # This is temporary - removing all metapages
            ###
            # df = df.drop(df[df.pagetitle.str.startswith('User') |
            #                 df.pagetitle.str.startswith('User talk') |
            #                 df.pagetitle.str.startswith('Talk')].index)

            # convert True/False to 1/0 respectively
            df['isReverted'] = df['isReverted'].astype(int)

            # df['isMetapage'] = df.apply(lambda row: check_metapage(row), axis=1)
            df['revtime'] = df.apply(lambda row: timestamp_to_millis(row, 'revtime'), axis=1)
            df['revertTime'] = df.apply(lambda row: timestamp_to_millis(row, 'revertTime'), axis=1)

            # split data into training and testing datasets (75%-25%)
            data_size = df.shape[0]
            split_point = int(math.ceil(data_size * 0.75))
            train_df, test_df = df[:split_point], df[split_point:]
            train_data_frame = train_data_frame.append(train_df, ignore_index=True)
            test_data_frame = test_data_frame.append(test_df, ignore_index=True)

# debug print
print "Size of train data set: ", train_data_frame.shape
print "Size of test data set: ", test_data_frame.shape

# calculate target vector on datasets
train_data_frame['isVandal'] = train_data_frame.apply(lambda row: detect_vandal_ntus(row), axis=1)
test_data_frame['isVandal'] = test_data_frame.apply(lambda row: detect_vandal_ntus(row), axis=1)

# sort data by nested sorting on pagetitle and revision time
train_data_frame.sort_values(['pagetitle', 'revtime'], ascending=True)

# write to out folder as csv files
train_data_frame.to_csv('out/train.csv', sep=',', encoding='utf-8', index=False)
test_data_frame.to_csv('out/test.csv', sep=',', encoding='utf-8', index=False)

end = time.time()

print "Time to preprocess data: ", math.ceil((end - start)/60), " minutes"

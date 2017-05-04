import matplotlib.pyplot as plt
import pandas as pd

wiki_processed_dataset = pd.read_csv('out/wikidata.csv', delimiter=',')
# wiki_processed_dataset = pd.read_csv('test_out/wiki_test.csv', delimiter=',')
print "total data size: ", wiki_processed_dataset.shape

# number of benign and vandal edits
benign_data_size = wiki_processed_dataset[wiki_processed_dataset['vandal'] == 0].shape[0]
print "number of benign rows: ", benign_data_size
vandal_data_size = wiki_processed_dataset[wiki_processed_dataset['vandal'] == 1].shape[0]
print "number of vandal rows: ", vandal_data_size


def analyze_feature(data, feature_name):
    print ""
    print "analysing feature: ", feature_name
    print data.shape
    benign_has_feature = data[(data['vandal'] == 0) & (data[feature_name] == 0)][feature_name].shape[0]
    print benign_has_feature
    vandal_has_feature = data[(data['vandal'] == 1) & (data[feature_name] == 1)][feature_name].shape[0]
    print vandal_has_feature
    print "percentage of benign having fm: ", 100.0 * float(benign_has_feature) / benign_data_size, "%"
    print "percentage of vandals having fm: ", 100.0 * float(vandal_has_feature) / vandal_data_size, "%"

# def analyze_feature_ntus(data, feature_name):
#     print ""
#     print "analysing feature: ", feature_name
#     print data.shape
#     benign_has_feature = data[(data['vandal'] == 0) & (data[feature_name] == -1)][feature_name].shape[0]
#     print benign_has_feature
#     vandal_has_feature = data[(data['vandal'] == 1) & (data[feature_name] == 1)][feature_name].shape[0]
#     print vandal_has_feature
#     print "percentage of benign having ntus: ", 100.0 * float(benign_has_feature) / benign_data_size, "%"
#     print "percentage of vandals having ntus: ", 100.0 * float(vandal_has_feature) / vandal_data_size, "%"
#


# bar_plot_data = wiki_processed_dataset[['ntus', 'fm', 'crmv', 'crmf' , 'crms', 'vandal']]
fm_data = wiki_processed_dataset[['fm', 'vandal']]
analyze_feature(fm_data, 'fm')
ntus_data = wiki_processed_dataset[['ntus', 'vandal']]
analyze_feature(ntus_data, 'ntus')
crmv_data = wiki_processed_dataset[['crmv', 'vandal']]
analyze_feature(crmv_data, 'crmv')
crmf_data = wiki_processed_dataset[['crmf', 'vandal']]
analyze_feature(crmf_data, 'crmf')
crms_data = wiki_processed_dataset[['crms', 'vandal']]
analyze_feature(crms_data, 'crms')

# print "analysing fm feature...."
# print fm_data.shape
# benign_has_fm = fm_data[(fm_data['vandal'] == 0) & (fm_data['fm'] == 0)]['fm'].shape[0]
# print benign_has_fm
# vandal_has_fm = fm_data[(fm_data['vandal'] == 1) & (fm_data['fm'] == 0)]['fm'].shape[0]
# print vandal_has_fm
# print "percentage of benign having fm: ", float(benign_has_fm) / benign_data_size
# print "percentage of vandals having fm: ", float(vandal_has_fm) / vandal_data_size

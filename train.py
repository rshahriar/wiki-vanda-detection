import cPickle
from collections import defaultdict
from sklearn import svm, preprocessing

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# dictionary for label encoder
d = defaultdict(LabelEncoder)

wiki_train_dataset = pd.read_csv('out/train.csv', delimiter=',')

# encode categorical features to numeric
encoder = LabelEncoder()
wiki_train_dataset['username'] = encoder.fit_transform(wiki_train_dataset['username'])
wiki_train_dataset['pagetitle'] = encoder.fit_transform(wiki_train_dataset['pagetitle'])

# convert dataset from pandas.dataframe to numpy.ndarray
X_train_dataset = wiki_train_dataset.as_matrix(columns=['username', 'revid', 'revtime', 'pagetitle'])
Y_train_dataset = wiki_train_dataset.as_matrix(columns=['isVandal'])

# scaling training/testing data
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(X_train_dataset)
X_train_dataset_transformed = scaler.transform(X_train_dataset)

# scaler = preprocessing.StandardScaler().fit(Y_train_dataset)
# Y_train_dataset_transformed = scaler.transform(Y_train_dataset)

# code for classifier
clf = svm.SVC(verbose=True)
clf.fit(X_train_dataset_transformed, Y_train_dataset.ravel())
# clf = tree.DecisionTreeClassifier()
# clf = RandomForestClassifier()
# clf = clf.fit(X_train_dataset, Y_train_dataset)

# STORE THE LEARNED AUTO-ENCODER, SO THAT YOU DONT HAVE TO LEARN IT EVERYTIME
FILE_LOCATION = "C:/Users/rakib/Desktop/Wiki Project/wiki_trained_model"
f = file(FILE_LOCATION + ".obj","wb") # Specify file location
cPickle.dump(clf, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

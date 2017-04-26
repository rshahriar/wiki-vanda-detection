import cPickle
from collections import defaultdict

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


# code for classifier
# svc_model = svm.SVC(kernel='linear', gamma=0.001, cache_size=5000, verbose=1)
# svc_model.fit(X_train_dataset, Y_train_dataset.ravel())
# clf = tree.DecisionTreeClassifier()
clf = RandomForestClassifier()
clf = clf.fit(X_train_dataset, Y_train_dataset)

# STORE THE LEARNED AUTO-ENCODER, SO THAT YOU DONT HAVE TO LEARN IT EVERYTIME
FILE_LOCATION = "C:/Users/rakib/Desktop/Wiki Project/wiki_trained_model"
f = file(FILE_LOCATION + ".obj","wb") # Specify file location
cPickle.dump(clf, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

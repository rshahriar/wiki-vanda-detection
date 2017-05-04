import cPickle
from collections import defaultdict

import math

import time
from sklearn import svm, preprocessing, tree

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# dictionary for label encoder
d = defaultdict(LabelEncoder)

wiki_train_dataset = pd.read_csv('out/wikidata.csv', delimiter=',')
# wiki_train_dataset = pd.read_csv('test_out/wiki_test.csv', delimiter=',')
print "UMD Wikipedia processed dataset size: ", wiki_train_dataset.shape

# encode categorical features to numeric
encoder = LabelEncoder()
wiki_train_dataset['username'] = encoder.fit_transform(wiki_train_dataset['username'])
wiki_train_dataset['pagetitle'] = encoder.fit_transform(wiki_train_dataset['pagetitle'])

# convert dataset from pandas.dataframe to numpy.ndarray
X_train_dataset = wiki_train_dataset.as_matrix(columns=['username', 'revtime', 'pagetitle',
                                                        'ntus', 'fm', 'crmv', 'crmf' , 'crms'])
Y_train_dataset = wiki_train_dataset.as_matrix(columns=['vandal'])
X_train, X_test, y_train, y_test = train_test_split(X_train_dataset, Y_train_dataset)

# scaling training data
# scaler = preprocessing.StandardScaler().fit(Y_train_dataset)
# Y_train_dataset_transformed = scaler.transform(Y_train_dataset)
scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(X_train)
X_train_dataset_transformed = scaler.transform(X_train)
X_test_dataset_transformed = scaler.transform(X_test)

# Initialize classifier
# clf = tree.DecisionTreeClassifier()
# clf = RandomForestClassifier()
# clf = svm.SVC(verbose=True)
# clf = MLPClassifier(hidden_layer_sizes=(8,8,8))
clf = LogisticRegression()
# clf = GaussianNB()

# Train the classifier
start = time.time()
# clf = clf.fit(X_train_dataset, Y_train_dataset)
clf = clf.fit(X_train_dataset_transformed, y_train.ravel())
end = time.time()
print "Time to train model: ", math.ceil((end - start)/60), " minutes"

# Predict on test data
start = time.time()
Y_predict_dataset = clf.predict(X_test_dataset_transformed)
end = time.time()
print "Time to predict test data: ", math.ceil((end - start)/60), " minutes"

# code for testing accuracy
print 'Accuracy score: ', accuracy_score(y_test, Y_predict_dataset)
print(confusion_matrix(y_test, Y_predict_dataset))
print(classification_report(y_test, Y_predict_dataset))


# STORE THE LEARNED AUTO-ENCODER, SO THAT YOU DONT HAVE TO LEARN IT EVERYTIME
FILE_LOCATION = "C:/Users/rakib/Documents/GitHub/wiki-vanda-detection/trained_models/trained_model"
f = file(FILE_LOCATION + ".obj","wb") # Specify file location
cPickle.dump(clf, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

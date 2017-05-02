import cPickle

import math
import pandas as pd
import time
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

FILE_LOCATION = "C:\Users\Rakib\Documents\GitHub\wiki-vanda-detection\trained_models\svc_trained_model"
print 'loading learned model...'
f = file(FILE_LOCATION +  ".obj","rb") # Specify file location
clf = cPickle.load(f)
f.close()
print 'loaded model..'

wiki_test_dataset = pd.read_csv('out/test.csv', delimiter=',')

# encode categorical features to numeric
encoder = LabelEncoder()
wiki_test_dataset['username'] = encoder.fit_transform(wiki_test_dataset['username'])
wiki_test_dataset['pagetitle'] = encoder.fit_transform(wiki_test_dataset['pagetitle'])

X_test_dataset = wiki_test_dataset.as_matrix(columns=['username', 'revid', 'revtime', 'pagetitle', 'ntus'])
Y_test_dataset = wiki_test_dataset.as_matrix(columns=['isVandal'])

# scaling testing data
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(X_test_dataset)
X_test_dataset_transformed = scaler.transform(X_test_dataset)

# Predict on test data
start = time.time()
Y_predict_dataset = clf.predict(X_test_dataset_transformed)
end = time.time()
print "Time to predict test data: ", math.ceil((end - start)/60), " minutes"

# code for testing accuracy
print 'Accuracy score: ', accuracy_score(Y_test_dataset, Y_predict_dataset)

# code for decoding
# wiki_train_dataset['username'] = encoder.inverse_transform(wiki_train_dataset['username'])
# print wiki_train_dataset.head(30)

# STORE THE LEARNED AUTO-ENCODER, SO THAT YOU DONT HAVE TO LEARN IT EVERYTIME
FILE_LOCATION = "C:\Users\Rakib\Documents\GitHub\wiki-vanda-detection\trained_models\svc_trained_model"
f = file(FILE_LOCATION + ".obj","wb") # Specify file location
cPickle.dump(clf, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

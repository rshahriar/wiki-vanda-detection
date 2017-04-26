# LOAD THE PREVIOUSLY LEARNED AUTO-ENCODER
import cPickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

FILE_LOCATION = "C:/Users/rakib/Desktop/Wiki Project/wiki_trained_model"
print 'loading learned model...'
f = file(FILE_LOCATION +  ".obj","rb") # Specify file location
wiki_trained_model = cPickle.load(f)
f.close()
print 'loaded model..'

wiki_test_dataset = pd.read_csv('out/test.csv', delimiter=',')

# encode categorical features to numeric
encoder = LabelEncoder()
wiki_test_dataset['username'] = encoder.fit_transform(wiki_test_dataset['username'])
wiki_test_dataset['pagetitle'] = encoder.fit_transform(wiki_test_dataset['pagetitle'])

X_test_dataset = wiki_test_dataset.as_matrix(columns=['username', 'revid', 'revtime', 'pagetitle'])
Y_test_dataset = wiki_test_dataset.as_matrix(columns=['isVandal'])

# Predict from classifier model
Y_predict_dataset = wiki_trained_model.predict(X_test_dataset)

print Y_test_dataset

# code for testing accuracy
print 'Accuracy store', accuracy_score(Y_test_dataset, Y_predict_dataset)

# code for decoding
# wiki_train_dataset['username'] = encoder.inverse_transform(wiki_train_dataset['username'])
# print wiki_train_dataset.head(30)


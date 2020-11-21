#from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

from Log import Log
import os
import sys


topdir = os.path.split(os.path.split(os.path.realpath(sys.argv[0]))[0])[0]
logfile = os.path.join(topdir, 'log/logistic_training.log')
logging = Log(logfile)


#Training
training_features = os.path.join(topdir, 'log/timeseries_training_value.csv')
training_labels = os.path.join(topdir, 'log/timeseries_training_target.csv')

# Training
df_numpy = np.genfromtxt(training_features, delimiter=",")
y_numpy = np.genfromtxt(training_labels, delimiter=",")

train_feature, test_feature, train_label, test_label = train_test_split(
    df_numpy, y_numpy, test_size=1/10.0, random_state=122)


scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_feature)
# Apply transform to both the training set and the test set.
train_feature = scaler.transform(train_feature)
test_feature = scaler.transform(test_feature)

model = LogisticRegression(solver = 'lbfgs', max_iter=200)
model.fit(train_feature, train_label)

# use the model to make predictions with the test data
y_pred = model.predict(test_feature)
# how did our model perform?
count_misclassified = (test_label != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(test_label, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))
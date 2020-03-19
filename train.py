"""
Training of data
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import func as fx
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifer, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import logger as lg
from sklearn.naive_bayes import *

data_file = './dataset.csv'

data = pd.read_csv(data_file, delimiter=',', header=None)
targets = data.iloc[:,[0]].values
inputs = data.iloc[:,1:].values

X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = LinearSVC().fit(X_train, y_train)
lg.success('LinearSVC: {:.2f}\n'.format(model.score(X_test, y_test)))

model = SVC().fit(X_train, y_train)
lg.success('SVC: {:.2f}\n'.format(model.score(X_test, y_test)))

model = KNeighborsClassifer().fit(X_train, y_train)
lg.success('KNN: {:.2f}\n'.format(model.score(X_test, y_test)))

model = RandomForestClassifier().fit(X_train, y_train)
lg.success('RandomForest: {:.2f}\n'.format(model.score(X_test, y_test)))

# testing naive bayesian classifiers
from sklearn.naive_bayes import *
model = GaussianNB()
model.fit(X_train, y_train)
lg.success('Gaussian NB: {:.2f}'.format(model.score(X_test, y_test)))

model = BernoulliNB()
model.fit(X_train, y_train)
lg.success('Bernoulli NB: {:.2f}'.format(model.score(X_test, y_test)))

model = MultinomialNB()
model.fit(X_train, y_train)
lg.success('Multinomial NB: {:.2f}'.format(model.score(X_test, y_test)))

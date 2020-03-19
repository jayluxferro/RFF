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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import logger as lg


data_file = './dataset.csv'

data = pd.read_csv(data_file, delimiter=',', header=None)
targets = data.iloc[:,[0]].values
inputs = data.iloc[:,1:].values
class_names = ['TX1', 'TX2']

def generateResults(random_state=20):
    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=random_state)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = LinearSVC(random_state=0).fit(X_train, y_train)
    print(model)
    lg.success('LinearSVC: {:.2f}\n'.format(model.score(X_test, y_test)))
    plot_confusion_matrix(model, X_test, y_test, normalize='true', display_labels=class_names, cmap=plt.cm.Blues)
    plt.title('Linear SVC - {:.2f}%'.format(model.score(X_test, y_test)))
    plt.show()


    model = SVC(random_state=0).fit(X_train, y_train)
    print(model)
    lg.success('SVC: {:.2f}\n'.format(model.score(X_test, y_test)))
    plot_confusion_matrix(model, X_test, y_test, normalize='true', display_labels=class_names, cmap=plt.cm.Blues)
    plt.title('SVC - {:.2f}%'.format(model.score(X_test, y_test)))
    plt.show()

    model = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
    print(model)
    lg.success('KNN: {:.2f}\n'.format(model.score(X_test, y_test)))
    plot_confusion_matrix(model, X_test, y_test, normalize='true', display_labels=class_names, cmap=plt.cm.Blues)
    plt.title('KNN - {:.2f}%'.format(model.score(X_test, y_test)))
    plt.show()

    model = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    print(model)
    lg.success('DecisionTree: {:.2f}\n'.format(model.score(X_test, y_test)))
    plot_confusion_matrix(model, X_test, y_test, normalize='true', display_labels=class_names, cmap=plt.cm.Blues)
    plt.title('Decision Tree - {:.2f}%'.format(model.score(X_test, y_test)))
    plt.show()

    model = LogisticRegression(random_state=0).fit(X_train, y_train)
    print(model)
    lg.success('LogisticRegression: {:.2f}\n'.format(model.score(X_test, y_test)))
    plot_confusion_matrix(model, X_test, y_test, normalize='true', display_labels=class_names, cmap=plt.cm.Blues)
    plt.title('Logistic Regression - {:.2f}%'.format(model.score(X_test, y_test)))
    plt.show()

# all params
#generateResults()

# only mag
"""
targets = data.iloc[:,[0]].values
inputs = data.iloc[:,1:5].values
generateResults(random_state=19)
"""
# only phase
targets = data.iloc[:,[0]].values
inputs = data.iloc[:,5:].values
generateResults(random_state=20)
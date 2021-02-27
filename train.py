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
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, confusion_matrix
import logger as lg

data_file = './dataset.csv'
data_path = './data/'
data = pd.read_csv(data_file, delimiter=',', header=0)
targets = data.iloc[:,[0]].values
inputs = data.iloc[:,1:].values

def generateResults(random_state=20, path='all'):
    results_path = data_path + path + '/'
    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=random_state)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = LinearSVC(random_state=0).fit(X_train, y_train)
    print(model)
    lg.success('LinearSVC: {:.2f}\n'.format(model.score(X_test, y_test)))
    y_pred = model.predict(X_test)
    fx.cm_analysis(y_test, y_pred, fx.class_names, results_path + 'LinearSVC_Detail.png')
    plot_confusion_matrix(model, X_test, y_test, normalize='true', display_labels=fx.class_names, cmap=plt.cm.Blues, include_values=True)
    plt.title('Linear SVC - {:.2f}'.format(model.score(X_test, y_test)))
    #plt.show()
    plt.savefig(results_path + 'LinearSVC.eps')
    plt.savefig(results_path + 'LinearSVC.png', dpi=1200)

    model = SVC(random_state=0).fit(X_train, y_train)
    print(model)
    lg.success('SVC: {:.2f}\n'.format(model.score(X_test, y_test)))
    y_pred = model.predict(X_test)
    fx.cm_analysis(y_test, y_pred, fx.class_names, results_path + 'SVC_Detail.png')
    plot_confusion_matrix(model, X_test, y_test, normalize='true', display_labels=fx.class_names, cmap=plt.cm.Blues, include_values=True)
    plt.title('SVC - {:.2f}'.format(model.score(X_test, y_test)))
    #plt.show()
    plt.savefig(results_path + 'SVC.eps')
    plt.savefig(results_path + 'SVC.png', dpi=1200)

    model = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
    print(model)
    lg.success('KNN: {:.2f}\n'.format(model.score(X_test, y_test)))
    y_pred = model.predict(X_test)
    fx.cm_analysis(y_test, y_pred, fx.class_names, results_path + 'KNN_Detail.png')
    plot_confusion_matrix(model, X_test, y_test, normalize='true', display_labels=fx.class_names, cmap=plt.cm.Blues, include_values=True)
    plt.title('KNN - {:.2f}'.format(model.score(X_test, y_test)))
    #plt.show()
    plt.savefig(results_path + 'KNN.eps')
    plt.savefig(results_path + 'KNN.png', dpi=1200)

    model = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    print(model)
    lg.success('DecisionTree: {:.2f}\n'.format(model.score(X_test, y_test)))
    y_pred = model.predict(X_test)
    fx.cm_analysis(y_test, y_pred, fx.class_names, results_path + 'DT_Detail.png')
    plot_confusion_matrix(model, X_test, y_test, normalize='true', display_labels=fx.class_names, cmap=plt.cm.Blues, include_values=True)
    plt.title('Decision Tree - {:.2f}'.format(model.score(X_test, y_test)))
    #plt.show()
    plt.savefig(results_path + 'DT.eps')
    plt.savefig(results_path + 'DT.png', dpi=1200)

    model = LogisticRegression(random_state=0).fit(X_train, y_train)
    print(model)
    lg.success('LogisticRegression: {:.2f}\n'.format(model.score(X_test, y_test)))
    y_pred = model.predict(X_test)
    fx.cm_analysis(y_test, y_pred, fx.class_names, results_path + 'LR_Detail.png')
    plot_confusion_matrix(model, X_test, y_test, normalize='true', display_labels=fx.class_names, cmap=plt.cm.Blues, include_values=True)
    plt.title('Logistic Regression - {:.2f}'.format(model.score(X_test, y_test)))
    #plt.show()
    plt.savefig(results_path + 'LR.eps')
    plt.savefig(results_path + 'LR.png', dpi=1200)

# all params
generateResults()

# only mag
targets = data.iloc[:,[0]].values
inputs = data.iloc[:,[1]].values
generateResults(path='mag', random_state=19)

# only phase
targets = data.iloc[:,[0]].values
inputs = data.iloc[:,[2]].values
generateResults(path='phase', random_state=20)

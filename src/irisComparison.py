__author__ = 'huseyincelik'

import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

import classifier


def vectorize(xArray):
    return xArray.reshape(xArray.shape[0], 1)

# Load data
iris = load_iris()

testCount, trainCount = 75, 75

features, target = iris.data, iris.target
# Parameters

for i in range(30):
    X = features
    y = target

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # Standardize
    # mean = X.mean(axis=0)
    # std = X.std(axis=0)
    # X = (X - mean) / std

    data = np.append(X[:trainCount], vectorize(y[:trainCount]), axis=1)

    # Train
    parent = classifier.Node(None)
    classifier.prepare_tree(data, parent)

    clf = DecisionTreeClassifier().fit(X[:trainCount], y[:trainCount])

    Zsingle = clf.predict(features[trainCount:])

    Zmulti = map(parent.classify, features[testCount:])
    print(sum(Zmulti != target[testCount:]), sum(Zsingle != target[testCount:]))


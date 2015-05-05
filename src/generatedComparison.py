__author__ = 'huseyincelik'

import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

import classifier


def vectorize(xArray):
    return xArray.reshape(xArray.shape[0], 1)


testRange, trainRange = range(0, 50), range(50, 100)

tryCount = 50

# Load data
multiErrors = np.zeros(tryCount)
singleErrors = np.zeros(tryCount)
for i in range(tryCount):
    features, target = make_classification(n_features=2, n_redundant=0, n_informative=2,
                                           n_clusters_per_class=1, n_classes=3, n_samples=100,
                                           class_sep=1.5)
    # Parameters
    np.random.seed(13)

    X = features
    y = target

    # Standardize
    # mean = X.mean(axis=0)
    # std = X.std(axis=0)
    # X = (X - mean) / std

    data = np.append(X[trainRange], vectorize(y[trainRange]), axis=1)

    # Train
    parent = classifier.Node(None)
    classifier.prepare_tree(data, parent)

    clf = DecisionTreeClassifier().fit(X[trainRange], y[trainRange])

    Zsingle = clf.predict(features[testRange])

    Zmulti = map(parent.classify, features[testRange])
    multiErrors[i] = sum(Zmulti != target[testRange])
    singleErrors[i] = sum(Zsingle != target[testRange])
    print(sum(Zmulti != target[testRange]), sum(Zsingle != target[testRange]))

print(np.mean(multiErrors), np.std(multiErrors))
print(np.mean(singleErrors), np.std(singleErrors))
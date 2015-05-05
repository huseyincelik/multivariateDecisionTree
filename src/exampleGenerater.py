__author__ = 'huseyincelik'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.datasets import make_classification

import classifier


def vectorize(xArray):
    return xArray.reshape(xArray.shape[0], 1)

# Load data
# iris = load_diabetes()
features, target = make_classification(n_features=2, n_redundant=0, n_informative=2,
                                       random_state=5, n_clusters_per_class=2, n_classes=2, n_samples=1000,
                                       class_sep=1.5)
# Parameters
np.random.seed(13)
n_classes = np.unique(target).__len__()
plot_colors = "by"
plot_step = 0.02

for pairidx, pair in enumerate([[0, 1]]):
    X = features
    y = target

    # Shuffle
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # Standardize
    # mean = X.mean(axis=0)
    # std = X.std(axis=0)
    # X = (X - mean) / std

    data = np.append(X, vectorize(y), axis=1)

    # Train
    parent = classifier.Node(None)

    classifier.prepare_tree(data, parent)

    # Plot the decision boundary

    x1 = X[:, 0]
    x2 = X[:, 1]

    x1_min, x1_max = x1.min() - 1, x1.max() + 1
    x2_min, x2_max = x2.min() - 1, x2.max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, plot_step),
                           np.arange(x2_min, x2_max, plot_step))

    contour_array = np.c_[xx1.ravel(), xx2.ravel()]

    Z = map(parent.classify, np.c_[xx1.ravel(), xx2.ravel()])
    Z = np.asarray(Z).reshape(xx1.shape)
    cs = plt.contourf(xx1, xx2, Z, cmap=plt.cm.Paired)

    plt.xlabel("feature " + str(pair[0]))
    plt.ylabel("feature " + str(pair[1]))
    plt.axis("tight")

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=str(i),
                    cmap=plt.cm.Paired)

plt.axis("tight")

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend()
plt.show()
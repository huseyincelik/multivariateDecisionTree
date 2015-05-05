__author__ = 'huseyincelik'

from sklearn.datasets import load_iris
import numpy as np

import classifier


def vectorize(xArray):
    return xArray.reshape(xArray.shape[0], 1)


iris = load_iris()

data = iris.data
target = iris.target


data = np.append(data, vectorize(target), axis=1)

parent = classifier.Node(None)

classifier.prepare_tree(data[:100], parent)

for each in data[100:]:
    print(parent.classify(each[:-1]), each[-1])

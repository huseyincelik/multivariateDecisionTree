import numpy as np
from scipy.spatial import distance

from Model import Tree, Node


def vectorize(array):
    return array.reshape(array.shape[0], 1)


def fit(X, y):
    tree = Tree()
    data = np.append(X, vectorize(y), axis=1)
    prepare_tree(data, tree.root)
    return tree


def prepare_tree(examples, node):
    classes = classes_of(examples)
    if classes.__len__() == 1:
        node.class_of = classes[0]
        return
    selected_class = classes[0]
    centroid = find_centroid(with_class(examples, selected_class))
    non_centroid = find_centroid(without_class(examples, selected_class))
    classified, non_classified = learn(examples, centroid, non_centroid)
    node.left = Node(centroid, parent=node)
    node.right = Node(non_centroid, parent=node)
    prepare_tree(classified, node.left)
    prepare_tree(non_classified, node.right)


def learn(examples, centroid, non_centroid):
    length = examples.__len__()
    centroid_bools = np.array(np.zeros(length), dtype=bool)
    for idx, example in enumerate(examples):
        cent_dist = distance.euclidean(example[:-1], centroid)
        non_cent_dist = distance.euclidean(example[:-1], non_centroid)
        centroid_bools[idx] = cent_dist <= non_cent_dist
    if centroid_bools.all() or (centroid_bools == False).all():
        centroid_bools[0] = centroid_bools[0] is False
    return examples[centroid_bools], examples[centroid_bools == False]


def with_class(examples, selected_class):
    return examples[examples[:, -1] == selected_class]


def without_class(examples, selected_class):
    return examples[examples[:, -1] != selected_class]


def find_centroid(examples):
    return np.mean(examples[:, :-1], axis=0)


def classes_of(examples):
    classes = np.unique(examples[:, -1])
    np.random.shuffle(classes)
    return classes

__author__ = 'huseyincelik'

import numpy as np
from scipy.spatial import distance


class Node:
    def __init__(self, centroid, left=None, right=None, parent=None, class_of=None):
        self.left = left
        self.right = right
        self.centroid = centroid
        self.class_of = class_of
        self.parent = parent

    def classify(self, test):
        if self.class_of is not None:
            return self.class_of
        left_dist = distance.euclidean(test, self.left.centroid)
        right_dist = distance.euclidean(test, self.right.centroid)
        return self.left.classify(test) if left_dist <= right_dist else self.right.classify(test)


def prepare_tree(examples, node):
    classes = classes_of(examples)
    if classes.__len__() == 1:
        node.class_of = classes[0]
        return
    selected_class = classes[0]
    selecteds = with_class(examples, selected_class)
    non_selected = without_class(examples, selected_class)
    centroid = find_centroid(selecteds)
    non_centroid = find_centroid(non_selected)
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


def with_class(examples, selectedClass):
    return examples[examples[:, -1] == selectedClass]


def without_class(examples, selectedClass):
    return examples[examples[:, -1] != selectedClass]


def find_centroid(examples):
    return np.mean(examples[:, :-1], axis=0)


def classes_of(examples):
    classes = np.unique(examples[:, -1])
    np.random.shuffle(classes)
    return classes




    # selectedClass = classes.get(0)
    # selecteds = examples.getWithClass(selectedClass)
    # nonSelecteds = examples.getWithoutClass(selectedClass)
    # selectedCentroid = findCentroid(selecteds)
    # nonSelectedCentroid = findCentroid(nonSelectedCentroid)
    # classified,nonClassified,node = classify(examples,selectedCentroid,nonSelectedCentroid)
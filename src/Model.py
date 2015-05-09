from scipy.spatial import distance


class Tree:
    def __init__(self):
        self.root = Node(None)
        self.depth = 0

    def predict(self, test):
        return self.root.classify(test)


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
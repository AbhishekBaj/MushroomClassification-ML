class Node:
    def __init__(self, value=None, feature=None, left=None, right=None, probabilities=None):
        self.value = value
        self.feature = feature
        self.left = left
        self.right = right
        self.probabilities = probabilities
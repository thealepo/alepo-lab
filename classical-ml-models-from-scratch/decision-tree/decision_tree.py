import numpy as np
from collections import Counter

def entropy(y):
    histogram = np.bincount(y)
    ps = histogram / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:
    def __init__(self , feature=None , threshold=None , left=None , right=None , * , value=None):
        self.feature = feature
        self.threshold = threshold
        self.left , self.right = left , right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self , min_samples_split=2 , max_depth=100 , n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self , X , y):
        # grow our tree
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features , X.shape[1])
        self.root = self._grow_tree(X , y)
    def _grow_tree(self , X , y , depth=0):
        n_samples , n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_val = self._most_common_label(y)
            return Node(value=leaf_val)

        feature_indices = np.random.choice(n_features , self.n_features , replace=False)

        # greedy search
        best_feat , best_thresh = self._best_criteria(X , y , feature_indices)
        left_indices , right_indices = self._split(X[:, best_feat] , best_thresh)

        left = self._grow_tree(X[left_indices , :] , y[left_indices] , depth+1)
        right = self._grow_tree(X[right_indices , :] , y[right_indices] , depth+1)
        return Node(best_feat , best_thresh , left , right)

    def _most_common_label(self , y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    def _best_criteria(self , X , y , feat_idxs):
        best_gain = -1
        split_idx , split_thresh = None , None

        for feature_index in feat_idxs:
            X_column = X[: , feature_index]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y , X_column , threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_thresh = threshold

        return split_index , split_thresh
    def _information_gain(self , y , X_column , split_thresh):
        # calculate parent Entropy
        parent_entropy = entropy(y)
        # generate a split
        left_indices , right_indices = self._split(X_column , split_thresh)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        # calculate weighted average of child Entropy
        n = len(y)
        n_left , n_right = len(left_indices) , len(right_indices)
        e_left , e_right = entropy(y[left_indices]) , entropy(y[right_indices])
        child_entropy = (n_left/n) * e_left + (n_right/n) * e_right

        # return ig
        ig = parent_entropy - child_entropy
        return ig

    def _split(self , X_column , split_thresh):
        left_indices = np.argwhere(X_column <= split_thresh).flatten()
        right_indices = np.argwhere(X_column > split_thresh).flatten()
        return left_indices , right_indices

    def predict(self , X):
        # traverse tree
        return np.array([self._traverse_tree(x , self.root) for x in X])
    def _traverse_tree(self , x , node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x , node.left)
        return self._traverse_tree(x , node.right)
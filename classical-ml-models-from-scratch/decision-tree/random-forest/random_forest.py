import numpy as np
from collections import Counter

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd() , '..')))
from decision_tree import DecisionTree

def bootstrapping(X , y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples , size=n_samples , replace=True)
    return X[indices] , y[indices]
def most_common_label(y):
    counter = Counter(y)
    return counter.most_common(1)[0][0]

class RandomForest:
    def __init__(self , n_trees=100 , min_samples_split=2 , max_depth=100 , n_features=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self , X , y):
        self.trees = []

        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features
            )

            X_sample , y_sample = bootstrapping(X , y)
            tree.fit(X_sample , y_sample)

            self.trees.append(tree)

    def predict(self , X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # 3 trees and 4 samples: [ [1111] , [0000] , [1111] ]
        # [ [101] , [101] , [101] , [101] ]
        tree_preds = np.swapaxes(tree_preds , 0 , 1)
        # 101 101 101 -> 1 1 1
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
        
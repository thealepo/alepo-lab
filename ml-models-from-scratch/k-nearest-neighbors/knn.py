import numpy as np
from collections import Counter

# normally in some utility class
def euclidean_distance(x1 , x2):  # can also be manhattan, minkowski
    return np.sqrt(np.sum((x1-x2)**2))
def manhattan_distance(x1 , x2):
    return np.sum(np.abs(x1-x2))
def minkowski_distance(x1 , x2 , exponent):
    return np.sum(np.abs(x1-x2)**exponent)**(1/exponent)

class KNN:
    def __init__(self , k=3 , distance_formula='euclidean' , power=2):
        self.k = k
        self.distance_formula = distance_formula
        self.power = power

    def fit(self , X , y):
        # no training step, just store training sample to use later
        self.X_train = X
        self.y_train = y

    def predict(self , X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    def _predict(self , x):
        # helper
        # calculate the distances and look at nearest neighbors, majority vote and choose most common class label

        # compute distances
        if self.distance_formula == 'euclidean':
            distances = [euclidean_distance(x,x_train) for x_train in self.X_train]
        elif self.distance_formula == 'manhattan':
            distances = [manhattan_distance(x,x_train) for x_train in self.X_train]
        elif self.distance_formula == 'minkowski':
            distances = [minkowski_distance(x,x_train,self.power) for x_train in self.X_train]
        else:
            distances = [euclidean_distance(x,x_train) for x_train in self.X_train]
        # k-nearest samples and labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # vote for most common class
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
import numpy as np

class Decision:
    def __init__(self):
        self.polarity = 1  #(-1 or +1)
        self.feature_index = None
        self.threshold = None
        self.alpha = None

    def predict(self , X):
        n_samples = X.shape[0]
        X_column = X[: , self.feature_index]

        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions


class AdaBoost:
    def __init__(self , n_classifiers = 5):
        self.n_classifiers = n_classifiers

    def fit(self , X , y):
        n_samples , n_features = X.shape

        # initialize weights
        w = np.full(n_samples , (1/n_samples))

        self.classifiers = []
        for _ in range(self.n_classifiers):
            classifier = Decision()

            min_error = float('inf')
            for i in range(n_features):
                X_column = X[: , i]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    missclassified_weights = w[y != predictions]
                    error = sum(missclassified_weights)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        min_error = error
                        classifier.polarity = p
                        classifier.threshold = threshold
                        classifier.feature_index = i

            EPSILON = 1e-10
            classifier.alpha = 0.5 * np.log( (1-error) / (error+EPSILON) )

            predictions = classifier.predict(X)

            w *= np.exp(-classifier.alpha * y * predictions)
            w /= np.sum(w)

            self.classifiers.append(classifier)
    
    def predict(self , X):
        classifier_predictions = [classifier.alpha * classifier.predict(X) for classifier in self.classifiers]
        return np.sign(np.sum(classifier_predictions , axis=0))

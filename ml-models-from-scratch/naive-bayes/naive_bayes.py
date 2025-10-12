import numpy as np

class NaiveBayes:

    def fit(self , X , y):
        n_samples , n_features = X.shape
        self._classes = np.unique(y)  # will find unique elements of an array
        n_classes = len(self._classes)

        # initialize mean , variance , priors
        self._mean = np.zeros((n_classes , n_features) , dtype=np.float64)
        self._var = np.zeros((n_classes , n_features) , dtype=np.float64)
        self._priors = np.zeros(n_classes , dtype=np.float64)

        for c in self._classes:
            X_c = X[c==y]
            self._mean[c,:] = X_c.mean(axis=0)
            self._var[c,:] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)  # how often class c occurs


    def predict(self , X):
        return [self._predict(x) for x in X]

    def _predict(self , x):
        # calculate posterior, class conditional, and prior
        # choose class w highest probability
        posteriors = []

        for i , c in enumerate(self._classes):
            prior = np.log(self._priors[i])
            class_conditional = np.sum(np.log(self._pdf(class_index=i , x=x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]
    def _pdf(self , class_index , x):
        mean , var = self._mean[class_index] , self._var[class_index]
        numerator , denominator = np.exp(- (x-mean)**2 / (2 * var)) , np.sqrt(2 * np.pi * var)
        return numerator / denominator
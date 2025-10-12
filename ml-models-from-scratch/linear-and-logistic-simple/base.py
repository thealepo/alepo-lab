import numpy as np

class Regression:
    def __init__(self , lr=0.001 , n_iterations=1000 , logistic=False):
        self.lr = lr
        self.n_iterations = n_iterations
        self.logistic = logistic
        self.w , self.b = None , None
    
    def _sigmoid(self , x):
        return 1 / (1 + np.exp(-x))

    def fit(self , X , y):
        n_samples , n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iterations):
            linear = np.dot(X , self.w) + self.b

            if self.logistic:
                y_pred = self._sigmoid(linear)
            else:
                y_pred = linear

            dw = (1 / n_samples) * np.dot(X.t , (y_pred-y))
            db = (1 / n_samples) * np.sum(y_pred-y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self , X):
        linear = np.dot(X , self.w) + self.b

        if self.logistic:
            y_pred = self._sigmoid(linear)
            return [1 if i > 0.5 else 0 for i in y_pred]
        else:
            y_pred = linear
            return y_pred

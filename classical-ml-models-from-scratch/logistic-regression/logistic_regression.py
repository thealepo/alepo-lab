import numpy as np

class LogisticRegression:
    def __init__(self , lr=0.001 , n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def fit(self , X , y):
        # training step and graident descent
        n_samples , n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iterations):
            linear = np.dot(X , self.w) + self.b
            y_pred = self._sigmoid(x=linear)

            dw = (1 / n_samples) * 2 * np.dot(X.T , (y_pred-y)) 
            db = (1 / n_samples) * np.sum(y_pred-y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self , X):
        linear = np.dot(X , self.w) + self.b
        y_pred = self._sigmoid(x=linear)
        classes = [1 if i > 0.5 else 0 for i in y_pred]

        return classes

    def _sigmoid(self , x):
        return 1 / (1 + np.exp(-(x)))
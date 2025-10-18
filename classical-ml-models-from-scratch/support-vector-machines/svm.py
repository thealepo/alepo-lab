import numpy as np

class SVM():
    def __init__(self , lr=0.001 , lambda_param=0.01 , n_iterations=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def fit(self , X , y):
        y_ = np.where(y <= 0 , -1 , 1)
        n_samples , n_features = X.shape

        self.w , self.b = np.zeros(n_features) , 0

        for _ in range(self.n_iterations):
            for i,x_i in enumerate(X):
                condition = y_[i] * (np.dot(x_i , self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                    self.b -= 0
                else:
                    self.w -= self.lr * ((2 * self.lambda_param * self.w) - np.dot(x_i , y_[i]))
                    self.b -= self.lr * y_[i]

    def predict(self , X):
        linear = np.dot(X , self.w) - self.b
        return np.sign(linear)

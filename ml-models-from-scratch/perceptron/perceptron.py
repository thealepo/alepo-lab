import numpy as np

class Perceptron:
    def __init__(self , lr=0.001 , n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.activation_function = self._unit_step_function
        self.w = None
        self.b = None

    def fit(self , X , y):
        n_samples , n_features = X.shape

        # initialize weights
        self.w , self.b = np.zeros(n_features) , 0

        y_ = np.array([1 if i >= 0 else 0 for i in y])

        for _ in range(self.n_iterations):
            for idx,x_i in enumerate(X):
                linear = np.dot(x_i , self.w) + self.b
                y_pred = self.activation_function(linear)

                update = self.lr * (y_[idx]-y_pred)
                self.w += update * x_i
                self.b += update * 1

    def predict(self , X):
        linear = np.dot(X , self.w) + self.b
        return self.activation_function(linear)

    def _unit_step_function(self , x):  # normally more complex (sigmoid, ReLU, tanh)
        return np.where(x >= 0 , 1 , 0)

    
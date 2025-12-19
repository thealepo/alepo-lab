from tensor import Tensor
import random

class Neuron:
    def __init__(self , nin):
        self.w = [Tensor(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Tensor(random.uniform(-1,1))

    def __call__(self , x):
        # w*x + b
        act = sum(wi*xi for wi,xi in zip(self.w , x)) + self.b
        out = act.tanh()
        return out

class Layer:
    def __init__(self , nin , nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self , x):
        out = [n(x) for n in self.neurons]
        return out

class MLP:
    def __init__(self , nin , nouts):
        size = [nin] + nouts
        self.layers = [Layer(size[i], size[i+1]) for i in range(nouts)]

    def __call__(self , x):
        for layer in self.layers:
            x = layer(x)
        return x


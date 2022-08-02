import numpy as np

class Network:
    def __init__(self, neurons_per_layer):
        self.neurons_per_layer  = neurons_per_layer
        self.num_layers         = len(neurons_per_layer)
        self.weights            = []
        self.biases             = []
        for i in range(self.num_layers):
            self.weights.append(np.zeros([]))
            self.biases.append(np.zeros(neurons_per_layer[i]))

        

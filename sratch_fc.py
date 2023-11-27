# scratch_fc.py
import numpy as np

class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.biases = np.zeros((output_size, 1))

    def forward(self, A):
        self.A_prev = A
        Z = np.dot(self.weights, A) + self.biases
        return Z

    def backward(self, dZ):
        m = self.A_prev.shape[1]
        dW = np.dot(dZ, self.A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.weights.T, dZ)
        return dA_prev, dW, db

    def update(self, dW, db, learning_rate):
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db

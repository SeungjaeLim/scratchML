# scratch_activation_function.py
import numpy as np

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(dA, Z):
    s = sigmoid(Z)
    return dA * s * (1 - s)

# train.py
import numpy as np
from scratch_cnn import ConvLayer
from scratch_fc import FullyConnectedLayer
from scratch_activation_function import relu, relu_derivative, sigmoid, sigmoid_derivative
from scratch_pooling import pool_forward, pool_backward

def forward_pass(X, parameters):
    # Retrieve parameters
    conv_layer, fc_layer = parameters

    # Convolutional Layer Forward
    Z1 = conv_layer.forward(X)
    A1 = relu(Z1)

    # Pooling Layer Forward
    P1, cache_pool = pool_forward(A1, f=2, stride=2, mode='max')

    # Flatten pooled output for feeding into fully connected layer
    F1 = P1.reshape(P1.shape[0], -1).T

    # Fully Connected Layer Forward
    Z2 = fc_layer.forward(F1)
    A2 = sigmoid(Z2)

    cache = (Z1, A1, P1, F1, Z2, A2, cache_pool)

    return A2, cache

def backward_pass(dA2, cache, parameters):
    # Retrieve cache and parameters
    Z1, A1, P1, F1, Z2, A2, cache_pool = cache
    conv_layer, fc_layer = parameters

    # Fully Connected Layer Backward
    dZ2 = sigmoid_derivative(dA2, Z2)
    dF1, dW2, db2 = fc_layer.backward(dZ2)

    # Reshape dF1 to match the shape of P1
    dP1 = dF1.T.reshape(P1.shape)

    # Pooling Layer Backward
    dA1 = pool_backward(dP1, cache_pool, mode='max')

    # ReLU Activation Backward
    dZ1 = relu_derivative(dA1, Z1)

    # Convolutional Layer Backward
    dA0, dW1, db1 = conv_layer.backward(dZ1)

    gradients = (dW1, db1, dW2, db2)

    return gradients

def update_parameters(parameters, gradients, learning_rate):
    conv_layer, fc_layer = parameters
    dW1, db1, dW2, db2 = gradients

    conv_layer.update(dW1, db1, learning_rate)
    fc_layer.update(dW2, db2, learning_rate)

# Initialize layers
conv_layer = ConvLayer(f=3, n_c=1, n_f=10, stride=1, pad=0)  # Example: 10 filters 
fc_layer = FullyConnectedLayer(input_size=9610, output_size=10)  # Example parameters

# Combine layers into a list (or use a more sophisticated structure if preferred)
parameters = [conv_layer, fc_layer]

# Example data 
X_train = np.random.randn(10, 64, 64, 3)  # 10 samples, 64x64 size, 3 channels
Y_train = np.random.randint(0, 2, (10, 1))  # Binary labels for this example

# Training loop
num_epochs = 10
learning_rate = 0.01

for epoch in range(num_epochs):
    # Forward pass
    A2, cache = forward_pass(X_train, parameters)

    # Compute cost 
    cost = np.mean(np.square(A2 - Y_train.T))  # Example: mean squared error

    # Backward pass
    dA2 = -(Y_train.T - A2)
    gradients = backward_pass(dA2, cache, parameters)

    # Update parameters
    update_parameters(parameters, gradients, learning_rate)

    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Cost: {cost}")

# Example test data 
X_test = np.random.randn(5, 64, 64, 3)  # 5 test samples, 64x64 size, 3 channels
Y_test = np.random.randint(0, 2, (5, 1))  # Binary labels for this example

def make_predictions(A2):
    predictions = A2 > 0.5
    return predictions

def calculate_accuracy(predictions, Y):
    correct_predictions = np.equal(predictions, Y.T)
    accuracy = np.mean(correct_predictions)
    return accuracy

# Test the trained model with test data
A2, _ = forward_pass(X_test, parameters)
predictions = make_predictions(A2)
accuracy = calculate_accuracy(predictions, Y_test)

print(f"Test Accuracy: {accuracy * 100}%")

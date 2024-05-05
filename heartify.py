import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Accuracy function
def accuracy(y, pred):
    count = np.sum(y == pred)
    return count * 100 / y.shape[0]

# Sigmoid function definition
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

# Softmax function for multiclass classification
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)

# Initializing the layer sizes, assuming the hidden layer has 4 units
def layer_sizes(X, Y, n_h):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    return n_x, n_h, n_y

# Initializing weights and biases for layer 1 and layer 2
# Initializing weights and biases for layer 1 and layer 2
def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01  # Update this line
    b2 = np.zeros((n_y, 1))                 # Update this line

    print("Shape of W1:", W1.shape)
    print("Shape of b1:", b1.shape)
    print("Shape of W2:", W2.shape)
    print("Shape of b2:", b2.shape)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# Forward propagation to compute activation values
def forward_propagation(X, parameters):
    W1, b1, W2, b2 = parameters.values()

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    print("Shape of Z2:", Z2.shape)
    print("Shape of A2:", A2.shape)
    print("Sample values of A2:", A2[:, :5])  # Print first 5 samples of A2

    return A2, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

# Softmax function for multiclass classification
def softmax(z):
    s = np.exp(z) / np.sum(np.exp(z), axis=0)
    print("Shape of softmax output:", s.shape)
    print("Sample values of softmax output:", s[:, :5])  # Print first 5 samples of softmax output
    return s


# Computes cost based on softmax function
def compute_cost(A2, Y):
    print("Shape of A2:", A2.shape)
    print("Shape of Y:", Y.shape)
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y)
    cost = -np.sum(logprobs) / m
    return float(cost)


# Computes gradients for learning
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1, W2 = parameters["W1"], parameters["W2"]
    A1, A2 = cache["A1"], cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

# Update parameters based on learning rate and gradients
def update_parameters(parameters, grads, learning_rate):
    for param in parameters:
        parameters[param] -= learning_rate * grads["d" + param]

    return parameters

# Model for NN
def nn_model(X, Y, n_h, num_iterations=1000, print_cost=False):
    n_x, _, n_y = layer_sizes(X, Y, n_h)
    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, 0.001)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters

# Predict function
def predict(parameters, X):
    A2, _ = forward_propagation(X, parameters)
    predictions = np.argmax(A2, axis=0) + 1
    return predictions

# Load data
X = np.loadtxt("C:\\Users\\Pari Singh Chauhan\\Downloads\\Prediction-Of-Cardiac-Arrhythmia-master\\Prediction-Of-Cardiac-Arrhythmia-master\\Neural networks\\reduced_features.csv", delimiter=",")
Y = np.loadtxt("C:\\Users\\Pari Singh Chauhan\\Downloads\\Prediction-Of-Cardiac-Arrhythmia-master\\Prediction-Of-Cardiac-Arrhythmia-master\\Neural networks\\target_output.csv", delimiter=",").astype(int)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# One hot encoding for multiclass
num_classes = 13
Y_train_encoded = np.eye(num_classes)[Y_train - 1]

# Transpose for correct shape
X_train, X_test = X_train.T, X_test.T
Y_train_encoded, Y_test = Y_train_encoded.T, Y_test.T

# Running the model
parameters = nn_model(X_train, Y_train_encoded, 70, num_iterations=1, print_cost=True)

# Predictions for testing set
predictions = predict(parameters, X_test)

# Accuracy of classification for testing set
print("Accuracy = %.2f %%" % accuracy(Y_test, predictions))
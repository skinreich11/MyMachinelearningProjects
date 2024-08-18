import random as random
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = [np.random.randn(layers[i], layers[i-1] + 1) for i in range(1, self.num_layers)]
        self.biases = [np.random.randn(layers[i], 1) for i in range(1, self.num_layers)]

    def sigmoid(self, z):
        return np.round(1 / (1 + np.exp(-z)), decimals=5)

    def sigmoid_derivative(self, z):
        return np.round(self.sigmoid(z) * (1 - self.sigmoid(z)), decimals=5)

    def feedforward(self, x):
        a = x
        for w, b in zip(self.weights,self.biases):
            z = np.dot(w, a) + b  # Adding bias input
            a = self.sigmoid(z)
        return a

    def backpropagate(self, x, y, lmbda=0):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # Forward pass
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            print(w, activation)
            z = np.dot(w, activation) + b  # Adding bias input
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_derivative(zs[-1])
        nabla_w[-1] = np.dot(delta, activations[-2].T)  # Adding bias input
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
        #nabla_b[-1] = delta
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_derivative(z)
            #delta = np.dot(self.weights[-l+1].T, delta)[:-1, :] * sp  # Removing bias contribution
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)  # Adding bias input
            nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
            #nabla_b[-l] = delta

        # Regularization
        if lmbda != 0:
            for i in range(len(nabla_w)):
                nabla_w[i] += (lmbda / len(x)) * self.weights[i]

        return nabla_w, nabla_b

    def train(self, training_data, epochs, learning_rate, lmbda=0, epsilon=1e-6):
        prev_cost = float('inf')
        for epoch in range(epochs):
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            for x, y in training_data:
                delta_nabla_w, delta_nabla_b = self.backpropagate(x, y, lmbda)
                nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                print(nabla_w, delta_nabla_w)
            self.weights = [w - (learning_rate / len(training_data)) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (learning_rate / len(training_data)) * nb for b, nb in zip(self.biases, nabla_b)]

            # Calculate current cost
            cost = 0
            for x, y in training_data:
                a = self.feedforward(x)
                cost += np.linalg.norm(a - y) ** 2
            cost /= (2 * len(training_data))
            print(cost)
            # Check for improvement in cost
            if prev_cost - cost < epsilon:
                print(f"Stopped at epoch {epoch+1} due to convergence.")
                break

            prev_cost = cost

    def cost_derivative(self, output_activations, y):
        return output_activations - y




x_train = [np.array([[0.13000]]), np.array([[0.42000]])]
y_train = [np.array([[0.90000]]), np.array([[0.23000]])]
training_data = list(zip(x_train, y_train))
initial_theta1 = np.array([[0.1], [0.2]])
initial_theta2 = np.array([[0.5, 0.6]])
weightsss = [initial_theta1, initial_theta2]

net = NeuralNetwork([1,2,1])
net.weights = np.array(weightsss)
net.biases = np.array([np.array([[0.4], [0.3]]), np.array([[0.7]])])
# Train the neural network
net.train(training_data.copy(), epochs=1, learning_rate=0.1, epsilon=0)
# Test the trained neural network
for x, y in training_data.copy():
    prediction = net.feedforward(x)
    print(f"Input: {x}, Target: {y}, Predicted: {prediction}")
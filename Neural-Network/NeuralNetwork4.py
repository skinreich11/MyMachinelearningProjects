import random
import numpy as np
# helpers
def sigmoid(z):
    #return np.round(1.0/(1.0+np.exp(-z)), decimals=5)
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    #return np.round(sigmoid(z)*(1-sigmoid(z)), decimals=5)
    return sigmoid(z)*(1-sigmoid(z))


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, learnRate, lmbda, test_data=None, epsilon=1e-6):
        samples = len(training_data)
        prev_cost = float('inf')
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                           for k in range(0, samples, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learnRate, lmbda)
            cost = 0
            for x, y in training_data:
                a = self.feedforward(x)
                cost += np.linalg.norm(a - y) ** 2
            cost /= (2 * len(training_data))
            if prev_cost - cost < epsilon:
                #print(f"Stopped at epoch {j+1} due to convergence.")
                break

            prev_cost = cost

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def backprop(self, x, y, lmbda):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for _layer in range(2, self.num_layers):
            z = zs[-_layer]
            sp = sigmoid_prime(z)
            print(self.weights[-_layer+1].T)
            print(delta)
            delta = np.dot(self.weights[-_layer+1].transpose(), delta) * sp
            nabla_b[-_layer] = delta
            nabla_w[-_layer] = np.dot(delta, activations[-_layer-1].transpose())
        if lmbda != 0:
            for i in range(len(nabla_w)):
                nabla_w[i] += (lmbda / len(x)) * self.weights[i]
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, learnRate, lmbda):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        instCosts = []
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y, lmbda)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(learnRate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learnRate/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(y[x]) for (x, y) in test_results)

x_train = [np.array([[0.13000]]), np.array([[0.42000]])]
y_train = [np.array([[0.90000]]), np.array([[0.23000]])]
training_data = list(zip(x_train, y_train))
initial_theta1 = np.array([[0.1], [0.2]])
initial_theta2 = np.array([[0.5, 0.6]])
weightsss = [initial_theta1, initial_theta2]

net = Network([1,2,1])
net.weights = np.array(weightsss)
net.biases = np.array([np.array([[0.4], [0.3]]), np.array([[0.7]])])
net.SGD(training_data.copy(), 1, 1, 0, 0)
instCosts = []
for x, y in training_data.copy():
    prediction = net.feedforward(x)
    instCost = np.sum((prediction - y) ** 2) / 2
    #print(f"inst cost: {instCost}")
    instCosts.append(instCost)
    print(f"Input: {x}, Target: {y}, Predicted: {prediction}")
#print(f"Total error: {np.sum(instCosts)}")
#print(f"Mean cost: {np.mean(instCosts)}")
x_train = [np.array([[0.32000], [0.68000]]), np.array([[0.83000], [0.02000]])]
y_train = [np.array([[0.75000], [0.98000]]), np.array([[0.75000], [0.28000]])]
training_data = list(zip(x_train, y_train))
initial_theta1 = np.array([[0.15000, 0.40000],
                           [0.10000, 0.54000],
                           [0.19000, 0.42000],
                           [0.35000, 0.68000]])
initial_theta2 = np.array([[0.67000, 0.14000, 0.96000, 0.87000],
                           [0.42000, 0.20000, 0.32000, 0.89000],
                           [0.56000, 0.80000, 0.69000, 0.09000]])
initial_theta3 = np.array([[0.87000, 0.42000, 0.53000],
                           [0.10000, 0.95000, 0.69000]])
weightsss = [initial_theta1, initial_theta2, initial_theta3]

net = Network([2, 4, 3, 2])
net.weights = np.array(weightsss)
net.biases = np.array([np.array([[0.42000], [0.72000], [0.01000], [0.30000]]),
                       np.array([[0.21000], [0.87000], [0.03000]]),
                       np.array([[0.04000], [0.17000]])])
net.SGD(training_data.copy(), 1, 1, 0, 0.25000)
for x, y in training_data.copy():
    prediction = net.feedforward(x)
    print(f"Input: {x}, Target: {y}, Predicted: {prediction}")



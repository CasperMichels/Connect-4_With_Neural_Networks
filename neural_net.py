import numpy


def sigmoid(x):
    x = numpy.clip(x, -700, 700)
    return 1 / (numpy.exp(-x) + 1)


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return numpy.tanh(x)


def tanh_prime(x):
    return 1 - numpy.power((numpy.tanh(x)), 2)


def leaky_relu(x):
    return numpy.where(x > 0, x, x * 0.01)


def leaky_relu_prime(x):
    return numpy.where(x > 0, 1, 0.01)


def cost(outputs, y):
    return numpy.mean(numpy.power(outputs - y, 2))


def cost_prime(outputs, y):
    return 2 * (outputs - y) / numpy.size(y)


class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None


class Dense(Layer):
    def __init__(self, n_inputs, n_neurons):
        super().__init__()
        self.weights = numpy.random.randn(n_neurons, n_inputs)
        self.biases = numpy.random.randn(n_neurons, 1)

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = numpy.dot(self.weights, self.inputs) + self.biases
        return self.outputs

    def backward(self, output_gradient, learning_rate):
        weights_gradient = numpy.average(numpy.dot(output_gradient, self.inputs.T), axis=0)
        bias_gradient = numpy.reshape(numpy.average(output_gradient, axis=1), (len(self.biases), 1))
        input_gradient = numpy.dot(self.weights.T, output_gradient)
        self.weights -= weights_gradient * learning_rate
        self.biases -= bias_gradient * learning_rate
        return input_gradient


class SigmoidActivation(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = sigmoid(self.inputs)
        return self.outputs

    def backward(self, output_gradient, learning_rate):
        input_gradient = numpy.multiply(output_gradient, sigmoid_prime(self.inputs))
        return input_gradient


class TanhActivation(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = tanh(self.inputs)
        return self.outputs

    def backward(self, output_gradient, learning_rate):
        input_gradient = numpy.multiply(output_gradient, tanh_prime(self.inputs))
        return input_gradient


class ReLUActivation(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = leaky_relu(self.inputs)
        return self.outputs

    def backward(self, output_gradient, learning_rate):
        input_gradient = numpy.multiply(output_gradient, leaky_relu_prime(self.inputs))
        return input_gradient


def train(neural_network, epochs, batch_size, learning_rate, inputs, full_desired_outputs):
    for e in range(epochs):
        epoch_cost = 0
        # randomise the data
        indices = numpy.arange(inputs.shape[0])
        numpy.random.shuffle(indices)
        inputs = inputs[indices]
        full_desired_outputs = full_desired_outputs[indices]
        # put data into batches
        outputs = inputs[0:batch_size]
        desired_outputs = full_desired_outputs[0:batch_size]
        # resize data for matrix multiplication
        outputs = numpy.reshape(outputs.T, (outputs.shape[1], batch_size))
        desired_outputs = numpy.reshape(desired_outputs.T, (desired_outputs.shape[1], batch_size))
        # forward propagation
        for layer in neural_network:
            outputs = layer.forward(outputs)
        epoch_cost += cost(outputs, desired_outputs)

        gradient = cost_prime(outputs, desired_outputs)
        # backward propagation
        for layer in reversed(neural_network):
            gradient = layer.backward(gradient, learning_rate)
        print(f"{epoch_cost / batch_size}")


X = numpy.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = numpy.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),
    SigmoidActivation(),
    Dense(3, 1),
    SigmoidActivation()
]

# train(network, 10000, 4, 1, X, Y)

# outputs = numpy.reshape(X.T, (2, X.shape[0]))
# for layer in network:
#     outputs = layer.forward(outputs)
# print(X, outputs)
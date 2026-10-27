import numpy as np

class Layer:
    def __init__(self, n, activation_function): #activation_function = None
        self.n = n
        self.weights = None
        self.inputs = None
        self.forward_pass = None
        #TODO: add functionality for none activation and elsing it to linear
        self.activation = activation_function

    def init_weights(self, input_dimension, rng):
        std = np.sqrt(2.0 / input_dimension) #He initialization added
        self.weights = rng.normal(0, std, (input_dimension + 1, self.n)).astype("float32")

    def forward(self, inputs):
        self.inputs = np.insert(inputs, 0, 1, axis=1)
        self.forward_pass = np.dot(self.inputs, self.weights)
        return self.activation.activation_func(self.forward_pass)

    def backward(self, lr, loss_gradient):
        loss_gradient = self.activation.backward(loss_gradient, self.forward_pass)
        next_gradient = np.dot(loss_gradient, np.transpose(self.weights[1:]))
        #TODO there is likey some scalar issues that need to be fixed
        self.weights -= lr * np.dot(np.transpose(self.inputs), loss_gradient)
        return next_gradient

class MLP:
    def __init__(self, input_n, loss, seed = 42):
        self.seed = seed
        self.input_n = input_n
        self.layers = []
        self.rng = np.random.default_rng(seed)
        self.loss = loss

    def add(self, layer: Layer):
        self.layers.append(layer)

    def initialize(self):
        self.rng = np.random.default_rng(self.seed)
        input_dimension = self.input_n

        for i in self.layers:
            i.init_weights(input_dimension, self.rng)
            input_dimension = i.n

    def forward(self, sample):
        if self.layers[0].weights is None:
            self.initialize()

        for layer in self.layers:
            sample = layer.forward(sample)

        return sample

    def backward(self, loss_gradient, lr):
        for layer in self.layers[::-1]:
            loss_gradient = layer.backward(lr, loss_gradient)

    def training_step(self, inputs, outputs, lr):
        y_hat = self.forward(inputs)
        loss = self.loss.activation_func(outputs, y_hat)
        loss_gradient = self.loss.derivative_func(outputs, y_hat)
        self.backward(loss_gradient, lr)

        return loss

    def train(self, input_data, output_data, batch_size, epochs, lr):
        for epoch in range(epochs):
            shuffle_permutation = np.random.permutation(len(input_data))
            shuffle_input = input_data[shuffle_permutation]
            shuffle_output = output_data[shuffle_permutation]

            for batch in range(int(len(shuffle_input)/batch_size)):
                print(self.training_step(shuffle_input[batch * batch_size:batch_size * (batch + 1)],shuffle_output[batch * batch_size:batch_size * (batch + 1)], lr))
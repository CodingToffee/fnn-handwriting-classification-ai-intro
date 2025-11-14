import math
import time

import numpy as np
import multiprocessing as mp


class FeedForwardNetwork:
    def __init__(self, init_weights=False):
        # initialize input weights for 784 input perceptrons to 200 hidden perceptrons
        self.weights_input = np.random.random((784, 200))
        # initialize hidden weights for 200 hidden perceptrons to 10 output perceptrons
        self.weights_hidden = np.random.random((200, 10))
        self.learning_rate = 0.03

        if init_weights:
            self.weights_input = np.load("weights_input.npy")
            self.weights_hidden = np.load("weights_hidden.npy")

    def sigmoid(self, x):
        x = np.clip(x, -709, 709)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def think(self, inputs):
        sigmoid_vectorized = np.vectorize(self.sigmoid)
        # calculate inputs for Hidden Layer
        inputs_hidden_layer = sigmoid_vectorized(np.dot(inputs, self.weights_input))
        # calculate inputs for Output Layer
        outputs = sigmoid_vectorized(np.dot(inputs_hidden_layer, self.weights_hidden))
        return inputs_hidden_layer, outputs

    def train(self, inputs, targets, iterations):
        sigmoid_derivative_vecotirzed = np.vectorize(self.sigmoid_derivative)

        for i in range(iterations):
            print(f"Iteration {i} of {iterations}")
            inputs_hidden_layer, outputs = self.think(inputs)
            # print(f"Output for Iteration {i}: {outputs}")
            # backpropagation
            e_out = np.subtract(targets, outputs)
            e_hidden = np.dot(e_out, self.weights_hidden.transpose())
            # print(f"Error for Iteration {i}: {e_out}\n")
            delta_w_ho = np.dot(inputs_hidden_layer.transpose(),
                                np.multiply(e_out, sigmoid_derivative_vecotirzed(outputs)))
            self.weights_hidden = self.weights_hidden + np.multiply(self.learning_rate, delta_w_ho)

            delta_w_ih = np.dot(inputs.transpose(),
                                np.multiply(e_hidden, sigmoid_derivative_vecotirzed(inputs_hidden_layer)))
            self.weights_input = self.weights_input + np.multiply(self.learning_rate, delta_w_ih)

        self.safe_weights_to_file()

    def train_parallel_2(self, inputs, targets, iterations):
        # Divide the data into chunks for multiprocessing
        num_processes = mp.cpu_count()
        chunk_size = len(inputs) // num_processes
        data_chunks = [(inputs[i:i + chunk_size], targets[i:i + chunk_size]) for i in
                       range(0, len(inputs), chunk_size)]

        sigmoid_derivative_vectorized = np.vectorize(self.sigmoid_derivative)
        for i in range(iterations):
            print(f"Iteration {i} of {iterations}")
            with mp.Pool(num_processes) as pool:
                results = pool.map(self.workload_parallelizable, data_chunks)

            # rebuild single results
            inputs_hidden_layer = np.concatenate([result[0] for result in results])
            outputs = np.concatenate([result[1] for result in results])
            e_out = np.concatenate([result[2] for result in results])
            e_hidden = np.concatenate([result[3] for result in results])

            # backpropagation
            delta_w_ho = np.dot(inputs_hidden_layer.transpose(),
                                np.multiply(e_out, sigmoid_derivative_vectorized(outputs)))
            self.weights_hidden = self.weights_hidden + np.multiply(self.learning_rate, delta_w_ho)

            delta_w_ih = np.dot(inputs.transpose(),
                                np.multiply(e_hidden, sigmoid_derivative_vectorized(inputs_hidden_layer)))
            self.weights_input = self.weights_input + np.multiply(self.learning_rate, delta_w_ih)

        self.safe_weights_to_file()

    def workload_parallelizable(self, data):
        inputs, targets = data
        inputs_hidden_layer, outputs = self.think(inputs)
        e_out = np.subtract(targets, outputs)
        e_hidden = np.dot(e_out, self.weights_hidden.transpose())

        return inputs_hidden_layer, outputs, e_out, e_hidden

    def train_parallel(self, inputs, targets, iterations):
        # Divide the data into chunks for multiprocessing
        num_processes = mp.cpu_count()
        chunk_size = len(inputs) // num_processes
        data_chunks = [(inputs[i:i + chunk_size], targets[i:i + chunk_size]) for i in
                       range(0, len(inputs), chunk_size)]

        # Create a multiprocessing Pool and train the network on each chunk of data
        with mp.Pool(num_processes) as pool:
            results = pool.map(self.train_sub_network, data_chunks)

        # Combine the results from each process
        weights_input = sum(result[0] for result in results) / num_processes
        weights_hidden = sum(result[1] for result in results) / num_processes

        # Update the weights of the network
        self.weights_input = weights_input
        self.weights_hidden = weights_hidden
        self.safe_weights_to_file()

    def train_sub_network(self, data_chunk):
        inputs, targets = data_chunk
        n = FeedForwardNetwork()
        n.train(inputs, targets, 1000)
        return n.weights_input, n.weights_hidden

    def safe_weights_to_file(self):
        np.save("weights_input", self.weights_input)
        np.save("weights_hidden", self.weights_hidden)


def prepare_data(filename):
    data_file = open(filename, "r")
    training_data_list = data_file.readlines()
    data_file.close()

    # normalize trainig data: 0-254 -> 0.01-0.99
    data_input = np.empty((0, 784))
    data_targets = []
    for record in training_data_list:
        data_input = np.vstack(
            (data_input, np.asfarray(record.split(",")[1:]) * 0.98 / 254.0 + 0.01))
        data_targets.append(int(record.split(",")[0]))

    # convert target data to np.array
    data_targets = np.array(data_targets)
    # convert target data to nx10 matrix with 0 for all values except for the target value which is 1
    data_targets_matrix = np.eye(10)[data_targets]

    return data_input, data_targets_matrix, data_targets


if __name__ == "__main__":
    training_data_input, training_data_targets_matrix, training_data_targets = prepare_data("mnist_train_full.csv")
    test_data_input, test_data_targets_matrix, test_data_targets = prepare_data("mnist_test_full.csv")
    print("Test data shape: ", test_data_input.shape)

    ffn = FeedForwardNetwork()
    # train the network
    print("Training")
    # measure time for training
    start = time.time()
    ffn.train(training_data_input, training_data_targets_matrix, 1000)
    end = time.time()
    print(f"Training took {end - start} seconds")

    # test the network
    # print(test_data_normalized.shape)
    print("Testing")
    inputs_hidden_layer, outputs = ffn.think(test_data_input)

    # bring outputs to target format
    print("Outputs raw: ", outputs)
    outputs = np.argmax(outputs, axis=1)
    print(outputs.shape)
    print("Comparing targets and outputs:")
    print(test_data_targets, outputs)

    # safe results to file
    np.save("test_data_targets", test_data_targets)
    np.save("outputs", outputs)

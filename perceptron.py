import math
import numpy as np


class Perceptron:
    def __init__(self):
        self.synaptic_weights = np.random.random(3)
        # self.synaptic_weights = [0.5, 0.5, 0.5]
        print(self.synaptic_weights)

    def sigmoid(self, x):
        return 1 / (1 + math.e ** -x)

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


    def think(self, inputs: np.array):
        """
        Calculation of the prediction
        :param inputs: x values
        :return: Predicted Value
        """
        sigmoid_vectorized = np.vectorize(self.sigmoid)
        return sigmoid_vectorized(np.dot(inputs, self.synaptic_weights))

    def train(self, inputs, targets, iterations):
        """
        Trains the perceptron with given inputs and targets for a number of iterations
        :param inputs: x values
        :param targets: y values that shall be predicted
        :param iterations: number of iterations
        :return: None
        """
        sigmoid_derivative_vectorized = np.vectorize(self.sigmoid_derivative)
        for i in range(iterations):
            # Let it think
            output = self.think(inputs)
            print(f"Output for Iteration {i}: {output}")

            # backpropagation
            # calculate the error
            e_out = targets - output
            print(f"Error for Iteration {i}: {e_out}\n")
            w_delta = np.dot((e_out * sigmoid_derivative_vectorized(output)), inputs)
            # adjust weights
            self.synaptic_weights = self.synaptic_weights + w_delta


if __name__ == "__main__":
    p = Perceptron()
    inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 1]], dtype=float)
    targets = np.array([0, 1, 1, 0])
    p.train(inputs, targets, 1000)
    # Get User Input for inputs
    I1 = int(input("Input 1: "))
    I2 = int(input("Input 2: "))
    I3 = int(input("Input 3: "))
    output = float(p.think(np.array([I1, I2, I3])))
    print(f"The prediction for {I1}, {I2}, {I3} is: {int(round(output))}")

import csv
import os

import numpy as np
import time


class FFNN:

    def __init__(self, sizes, l_rate=0.01):
        self.sizes = sizes
        self.l_rate = l_rate

        # we save all parameters in the neural network in this dictionary
        self.params = self.initialization()

    def initialization(self):
        # number of nodes in each layer
        input_layer = self.sizes[0]
        hidden_1 = self.sizes[1]
        hidden_2 = self.sizes[2]
        output_layer = self.sizes[3]

        params = {
            'W1': np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2': np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3': np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

        return params

    def sigmoid(self, x, derivative=False):
        temp_sig = 1 / (1 + np.exp(-x))
        if derivative:
            return temp_sig * (1 - temp_sig)
        return temp_sig

    def softmax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - np.max(x))
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def relu(self, x, derivative=False):
        x[x > 0] = 1
        x[x <= 0] = 0
        return x

    def predict(self, x_predict):
        params = self.params

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.relu(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.relu(params['Z3'])

        return params['A3']

    def backward_pass(self, y_train, output):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is
                  caused  by the dot and multiply operations on the huge arrays.

                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        params = self.params
        change_w = {}

        # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.sigmoid(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.relu(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.relu(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def train(self, x_train, y_train, x_val, y_val, epochs=1):
        start_time = time.time()
        for iteration in range(epochs):
            print("Total iterations: " + str(len(x_train)))
            counter = 0
            for x, y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)

            error = self.compute_error(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Mean Error: {2}'.format(
                iteration + 1, time.time() - start_time, error
            ))

    def update_network_parameters(self, changes_to_w):
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.

            θ = θ - η * ∇J(x, y),
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        '''

        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_error(self, x_val, y_val):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        error = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            total_error = (abs(output[0] - y[0]) + abs(output[1] - y[1])) / 2
            error.append(total_error)

        return np.mean(error)


if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, 'samples.csv')
    csv_file = open(abs_file_path, 'r')
    set_complete = np.zeros((151, 4))
    index = 0
    for x_1, x_2, y_1, y_2 in csv.reader(csv_file, delimiter=','):
        set_complete[index][0] = x_1
        set_complete[index][1] = x_2
        set_complete[index][2] = y_1
        set_complete[index][3] = y_2
        index += 1

    train_set = set_complete[:120, :]
    test_set = set_complete[120:151, :]
    train_X = train_set[:, 0:2]
    train_y = train_set[:, 2:4]
    test_X = test_set[:, 0:2]
    test_y = test_set[:, 2:4]

    train_X = (train_X / 300.0).astype('float32')
    test_X = (test_X / 300.0).astype('float32')

    ffnn = FFNN(sizes=[2, 1, 1, 2])
    ffnn.train(train_X, train_y, test_X, test_y)
    for i in range(50):
        np.random.shuffle(set_complete)

        train_set = set_complete[:140, :]
        test_set = set_complete[140:151, :]
        train_X = train_set[:, 0:2]
        train_y = train_set[:, 2:4]
        test_X = test_set[:, 0:2]
        test_y = test_set[:, 2:4]

        train_X = (train_X / 300).astype('float32')
        test_X = (test_X / 300).astype('float32')

        ffnn.train(train_X, train_y, test_X, test_y)

    print(ffnn.forward_pass([150, 150]))

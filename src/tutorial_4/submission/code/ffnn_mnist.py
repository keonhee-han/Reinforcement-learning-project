from mnist import MNIST
import numpy as np
import time


class FFNN:

    def __init__(self, sizes, l_rate=0.001):
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
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']

    def backward_pass(self, y_train, output):
        params = self.params
        change_w = {}

        # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
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
                if counter % 10000 == 0:
                    print(counter)
                counter += 1

            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2}'.format(
                iteration + 1, time.time() - start_time, accuracy
            ))

    def update_network_parameters(self, changes_to_w):
        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)


if __name__ == '__main__':
   
    test = MNIST('python-mnist/data', return_type='numpy')
    train_X, train_y = test.load_training()

    test_X, test_y = test.load_testing()

    print(train_X.shape)
    print(train_y.shape)
    train_set = np.column_stack((train_X, train_y))
    test_set = np.column_stack((test_X, test_y))
    set_complete = np.concatenate((train_set, test_set), axis=0)

    temp = np.zeros((train_y.shape[0], 10))

    index = 0
    for val in train_y:
        temp[index][int(val)] = 1
        index += 1
    train_y = temp
    temp = np.zeros((test_y.shape[0], 10))

    index = 0
    for val in test_y:
        temp[index][val] = 1
        index += 1
    test_y = temp

    train_X = (train_X / 255.0).astype('float32')
    test_X = (test_X / 255.0).astype('float32')

    ffnn = FFNN(sizes=[784, 128, 64, 10])
    ffnn.train(train_X, train_y, test_X, test_y)
    for i in range(30):

        np.random.shuffle(set_complete)

        train_set = set_complete[:60000, :]
        test_set = set_complete[60000:70000, :]
        train_X = train_set[:, :-1]
        train_y = train_set[:, -1]
        test_X = test_set[:, :-1]
        test_y = test_set[:, -1]

        temp = np.zeros((train_y.shape[0], 10))

        index = 0
        for val in train_y:
            temp[index][int(val)] = 1
            index += 1
        train_y = temp
        temp = np.zeros((test_y.shape[0], 10))

        index = 0
        for val in test_y:
            temp[index][val] = 1
            index += 1
        test_y = temp

        train_X = (train_X / 255.0).astype('float32')
        test_X = (test_X / 255.0).astype('float32')

        ffnn.train(train_X, train_y, test_X, test_y)

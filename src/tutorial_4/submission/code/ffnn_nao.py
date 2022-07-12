import csv
import os

import numpy as np
import time


# Implmentation of a MLP FFNN for execution on nao
class FFNN:

    def __init__(self, sizes, l_rate=0.01):
        self.sizes = sizes
        self.l_rate = l_rate

        # we save all parameters in the neural network in this dictionary
        self.params = self.initialization()
        self.max_pitch = 0
        self.max_roll = 0
        self.min_pitch = 0
        self.min_roll = 0
    

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

    # Function for saving the weights and parameters after training
    def load_weights(self):
        with open('weights_nao.npy', 'rb') as f:
            self.params['W1'] = np.load(f)
            self.params['W2'] = np.load(f)
            self.params['W3'] = np.load(f)
            self.params['MAX_P'] = np.load(f)
            self.params['MIN_P'] = np.load(f)
            self.params['MAX_R'] = np.load(f)
            self.params['MIN_R'] = np.load(f)


    # standalone function which loads the weights saved during training for use on nao
    def forward_prediction(self, x_train):
        x_train = x_train / 300.0
        x_train = x_train.reshape(2,1)
        self.load_weights()
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train
        params['Z1'] = np.dot(params['W1'], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])
        params['Z2'] = np.dot(params['W2'], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])
       
        params['Z3'] = np.dot(params['W3'], params['A2'])
        params['A3'] = params['Z3']
        
        result = params['A3']
       
        result[0]= result[0] * (abs(self.params['MIN_P'])- abs(self.params['MAX_P'])) - abs(self.params['MIN_P'])
       
        result[1] = result[1] * (abs(self.params['MIN_R'])+ abs(self.params['MAX_R'])) - abs(self.params['MIN_R'])

        return result


    # Forward pass
    def forward_pass(self, x_train):
        params = self.params

        params['A0'] = x_train
        
        # Sigmoid activation function
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])
        
        # SIgmoid activation function
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])

        # Linear activation function
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = params['Z3']
        return params['A3']

    # Backward propagation
    def backward_pass(self, y_train, output):
        params = self.params
        change_w = {}

        # Calculate W3 update
        # Linear activation function
        error = 2 * (output - y_train) / output.shape[0] #* self.relu(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        # Sigmoid activation function
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        # Sigmoid activation function
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    # Does the training: forward pass, backward pass & update of network parameters
    # Also prints the error and defines the number of training epochs
    def train(self, x_train, y_train, x_val, y_val, epochs=10000):
        start_time = time.time()
        for iteration in range(epochs):
           
            counter = 0
            for x, y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)

            error = self.compute_error(x_val, y_val)
            print ('Epoch: {0}, Time Spent: {1:.2f}s, Mean Error: {2}'.format(
                iteration + 1, time.time() - start_time, error
            ))

    def update_network_parameters(self, changes_to_w):
        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_error(self, x_val, y_val):
        error = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            total_error = (abs(output[0] - y[0]) + abs(output[1] - y[1])) / 2
            error.append(total_error)

        return np.mean(error)


if __name__ == '__main__':
    # read in samples (here: not normalized, happens later)
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
    
    # get max & min shoulder pitch & roll of training set
    max_pitch = np.amax(train_y[:,0])
    min_pitch = np.amin(train_y[:,0])

    max_roll = np.amax(train_y[:,1])
    min_roll = np.amin(train_y[:,1])
   
    # normalize shoulder pitch & roll 
    train_y[:,0] = ((train_y[:,0] + abs(min_pitch))/ (abs(min_pitch)- abs(max_pitch))).astype('float32')
    train_y[:,1] = ((train_y[:,1] + abs(min_roll))/ (abs(min_roll)+ abs(max_roll))).astype('float32')
    
    test_y[:,0] = ((test_y[:,0] + abs(min_pitch))/ (abs(min_pitch)- abs(max_pitch))).astype('float32')
    test_y[:,1] = ((test_y[:,1] + abs(min_roll))/ (abs(min_roll)+ abs(max_roll))).astype('float32')
    
    # Initialize NN with 2 hidden layers with 64 neurons
    ffnn = FFNN(sizes=[2, 64, 64, 2])
    ffnn.train(train_X, train_y, test_X, test_y)

    # saves trained weights and values for denormalization 
    with open('weights_nao.npy', 'wb') as f:
        np.save(f, ffnn.params["W1"])
        np.save(f, ffnn.params["W2"])
        np.save(f, ffnn.params["W3"])
        
        np.save(f, max_pitch)
        np.save(f, min_pitch)
        np.save(f, max_roll)
        np.save(f, min_roll)


        
   
  
    



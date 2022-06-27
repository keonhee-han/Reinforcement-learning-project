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
        self.max_pitch = 0
        self.max_roll = 0
        self.min_pitch = 0
        self.min_roll = 0
        self.w1 = []
        self.w2 = []
        self.w3 = []

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


    def load_weights(self):
        with open('/home/bio/Desktop/BIHR_Nao2022/src/tutorial_4/scripts/weights_nao.npy', 'rb') as f:
            self.w1 = np.load(f)
            self.w2 = np.load(f)
            self.w3 = np.load(f)
            self.max_pitch = np.load(f)
            self.min_pitch = np.load(f)
            self.max_roll = np.load(f)
            self.min_roll = np.load(f)


    def forward_prediction(self, x_train):
        #x_train = x_train / 300
        x_train = x_train.reshape(2,1)
        self.load_weights()
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train
        #print("A0",params['A0'])
        # input layer to hidden layer 1
        params['Z1'] = np.dot(self.w1, params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])
        #print("A1", params['A1'])
        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(self.w2, params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])
        #print("A2", params['A2'])
        # hidden layer 2 to output layer
        params['Z3'] = np.dot(self.w3, params['A2'])
        params['A3'] = params['Z3']
        #print("A3", params['A3'])
        #self.relu(params['Z3'])
        #print(params['A3'])
        result = params['A3']
        #print("-----",result)
        #print(abs(self.min_pitch))
        #print(result[0])
        result[0]= result[0] * (abs(self.min_pitch)- abs(self.max_pitch)) - abs(self.min_pitch)
        # normalize shoulder roll
        result[1] = result[1] * (abs(self.min_roll)+ abs(self.max_roll)) - abs(self.min_roll)

        return result

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train
        #print("A0:",params['A0'] )
        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])
        #print("A1:", params['A1'])
        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])
        #print("A2:", params['A2'])
        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = params['Z3']
        #print("A3:", params['A3'])
        #self.relu(params['Z3'])
        #print(params['A3'])
        return params['A3']

    def backward_pass(self, y_train, output):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            For the output layer, no derivative is used since we use a linear activation function
        '''
        params = self.params
        change_w = {}

        # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] #* self.relu(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def train(self, x_train, y_train, x_val, y_val, epochs=10000):
        start_time = time.time()
        for iteration in range(epochs):
            #print("Total iterations: " + str(len(x_train)))
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

    # normalize data in range [0,1]
    max_pix_x = np.amax(train_X[:,0])
    max_pix_y = np.amax(train_X[:,1])

    min_pix_x = np.amin(train_X[:,0])
    min_pix_y = np.amin(train_X[:,1])

    train_X = (train_X / 300.0).astype('float32')
    test_X = (test_X / 300.0).astype('float32')
    #train_X[:,0] = (train_X[:,0] - min_pix_x) / (max_pix_x - min_pix_x)
    #train_X[:,1] = (train_X[:,1] - min_pix_y) / (max_pix_y - min_pix_y)

    #test_X[:,0] = (test_X[:,0] - min_pix_x) / (max_pix_x - min_pix_x)
    #test_X[:,1] = (test_X[:,1] - min_pix_y) / (max_pix_y - min_pix_y)
    print(train_X)

    # normalize shoulder pitch
    max_pitch = np.amax(train_y[:,0])
    min_pitch = np.amin(train_y[:,0])

    max_roll = np.amax(train_y[:,1])
    min_roll = np.amin(train_y[:,1])
    print(max_pitch, min_pitch, max_roll, min_roll)
    train_y[:,0] = ((train_y[:,0] + abs(min_pitch))/ (abs(min_pitch)- abs(max_pitch))).astype('float32')
    # normalize shoulder roll
    train_y[:,1] = ((train_y[:,1] + abs(min_roll))/ (abs(min_roll)+ abs(max_roll))).astype('float32')
    
    test_y[:,0] = ((test_y[:,0] + abs(min_pitch))/ (abs(min_pitch)- abs(max_pitch))).astype('float32')
    test_y[:,1] = ((test_y[:,1] + abs(min_roll))/ (abs(min_roll)+ abs(max_roll))).astype('float32')
    
    ffnn = FFNN(sizes=[2, 64, 64, 2])
    ffnn.train(train_X, train_y, test_X, test_y)
    with open('/home/bio/Desktop/BIHR_Nao2022/src/tutorial_4/scripts/weights_nao.npy', 'wb') as f:
        np.save(f, ffnn.params["W1"])
        np.save(f, ffnn.params["W2"])
        np.save(f, ffnn.params["W3"])
        # values for denormalization
        np.save(f, max_pitch)
        np.save(f, min_pitch)
        np.save(f, max_roll)
        np.save(f, min_roll)

        
    def denormalize(result):

        result[0] = result[0] * (abs(min_pitch)- abs(max_pitch)) - abs(min_pitch)
        result[1] = result[1] * (abs(min_roll)+ abs(max_roll)) - abs(min_roll)
        print(result)
        return result 
        
    #print(ffnn.forward_pass([150, 150]))
    print("test result: ")

    result = ffnn.forward_pass([119.0/300.0, 139.0/300.0])
    denormalize(result)

    array = np.array([119.0/300.0, 139.0/300.0])
    result = ffnn.forward_prediction(array)

    #76,213,-0.5951499938964844,0.21778607368469238
    #108,179,-0.6825881004333496,0.17943596839904785
    #119,139,-0.817579984664917,0.15949392318725586
    #95,85,-0.9172899723052979,0.2009119987487793
    #66,82,-0.9387660026550293,0.2668740749359131
    print(result)

    



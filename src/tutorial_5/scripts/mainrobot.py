import csv
import pandas as pd
import numpy as np
import math

maxY1 = None
minY1 = None
maxY2 = None
minY2 = None
maxX1 = None
minX1 = None
maxX2 = None
minX2 = None

# followed this website : https://python.plainenglish.io/how-to-de-normalize-and-de-standardize-data-in-python-b4600cf9ee6
def norm(data,min,max):
    norm = (data-min)/(max-min)
    return np.asarray(norm)

def denorm(norm,min,max):
    var = (norm*(max-min)+max)
    return np.asarray(norm)

class NeuralNetwork:
    def __init__(self, X, y, batch = 64, lr = 5e-5,  epochs = 500):
        self.input = X 
        self.target = y
        self.batch = batch
        self.epochs = epochs
        self.lr = lr
        
        self.x = [] # batch input 
        self.y = [] # batch target value
        self.loss = []
        self.acc = []
        
        self.init_weights()
      
    def init_weights(self):
        self.W1 = np.random.randn(self.input.shape[0],256)
        self.W2 = np.random.randn(self.W1.shape[1],128)
        self.W3 = np.random.randn(self.W2.shape[1],self.target.shape[0])

        self.b1 = np.random.randn(self.W1.shape[1],)
        self.b2 = np.random.randn(self.W2.shape[1],)
        self.b3 = np.random.randn(self.W3.shape[1],)
    def ReLU(self, x):
        return np.maximum(0,x)

    def dReLU(self,x):
        return 1 * (x > 0) 

    def softmax(self, z):
        z = z - np.max(z, axis = 1).reshape(z.shape[0],1)
        return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(z.shape[0],1)

    def feedforward(self):
        print(self.x.shape[0])
        print(self.W1.shape[0])
        assert self.x.shape[0] == self.W1.shape[0]
        self.z1 = self.x.dot(self.W1) + self.b1
        self.a1 = self.ReLU(self.z1)
        assert self.a1.shape[1] == self.W2.shape[0]
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = self.ReLU(self.z2)
        assert self.a2.shape[1] == self.W3.shape[0]
        self.z3 = self.a2.dot(self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        self.error = self.a3 - self.y

    def backprop(self):
        dcost = (1/self.batch)*self.error

        DW3 = np.dot(dcost.T,self.a2).T
        DW2 = np.dot((np.dot((dcost),self.W3.T) * self.dReLU(self.z2)).T,self.a1).T
        DW1 = np.dot((np.dot(np.dot((dcost),self.W3.T)*self.dReLU(self.z2),self.W2.T)*self.dReLU(self.z1)).T,self.x).T

        db3 = np.sum(dcost,axis = 0)
        db2 = np.sum(np.dot((dcost),self.W3.T) * self.dReLU(self.z2),axis = 0)
        db1 = np.sum((np.dot(np.dot((dcost),self.W3.T)*self.dReLU(self.z2),self.W2.T)*self.dReLU(self.z1)),axis = 0)

        assert DW3.shape == self.W3.shape
        assert DW2.shape == self.W2.shape
        assert DW1.shape == self.W1.shape

        assert db3.shape == self.b3.shape
        assert db2.shape == self.b2.shape
        assert db1.shape == self.b1.shape 

        self.W3 = self.W3 - self.lr * DW3
        self.W2 = self.W2 - self.lr * DW2
        self.W1 = self.W1 - self.lr * DW1

        self.b3 = self.b3 - self.lr * db3
        self.b2 = self.b2 - self.lr * db2
        self.b1 = self.b1 - self.lr * db1

    def shuffle(self):
        idx = [i for i in range(self.input.shape[0])]
        np.random.shuffle(idx)
        self.input = self.input[idx]
        self.target = self.target[idx]
    
    def train(self):
        for epoch in range(self.epochs):
            l = 0
            acc = 0
            self.shuffle()

            for batch in range(self.input.shape[0]//self.batch-1):
                start = batch*self.batch
                end = (batch+1)*self.batch
                self.x = self.input[start:end]
                self.y = self.target[start:end]
                self.feedforward()
                self.backprop()
                l+=np.mean(self.error**2)
                acc+= np.count_nonzero(np.argmax(self.a3,axis=1) == np.argmax(self.y,axis=1)) / self.batch
            self.loss.append(l/(self.input.shape[0]//self.batch))
            self.acc.append(acc*100/(self.input.shape[0]//self.batch))
            print("Current Loss :",l/(self.input.shape[0]//self.batch))
            print("Current acc : ", acc*100/(self.input.shape[0]//self.batch))





def main():
    # This part just read CSV data and send to CMAC FUNCTION
    csvData = pd.read_csv('samples.csv', sep=',',header=None)
    csvData  = csvData.to_numpy()
    Y1 = np.asarray(csvData[:,0])
    Y2 = np.asarray(csvData[:,1])
    X1 = np.asarray(csvData[:,2])
    X2 = np.asarray(csvData[:,3])
    # These Parts are defined for data normalization
    maxY1 = max(Y1)
    maxY2 = max(Y2)
    maxX1 = max(X1)
    maxX2 = max(X2)
    minY1 = min(Y1)
    minY2 = min(Y2)
    minX1 = min(X1)
    minX2 = min(X2)
    normY1 = np.asarray(norm(Y1[:],maxY1,minY1))
    normY2 = np.asarray(norm(Y2[:],maxY2,minY2))
    normX1 = np.asarray(norm(X1[:],maxX1,minX1))
    normX2 = np.asarray(norm(X2[:],maxX2,minX2))
    X = np.asanyarray([normX1, normX2])
    Y = np.asanyarray([normY1, normY2])
    NN = NeuralNetwork(X,Y)
    NN.train()




if __name__ == '__main__':
    main()
    
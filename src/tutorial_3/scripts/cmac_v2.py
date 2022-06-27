#!/usr/bin/env python
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
RESOLUTION_RECEPTIVE = 50
RECEPTIVE_FIELD_SIZE = 3 # n_a
DIM_INPUT = 2
DIM_OUTPUT = 2
LEARNING_RATE = 0.1
TRAINING_ITERATIONS = 1000
"""
d = list(np.empty([2, RECEPTIVE_FIELD_SIZE]))
d[0][0] = 2
d[0][1] = 3
d[0][2] = 1
d[1][0] = 3
d[1][1] = 2
d[1][2] = 1
weights = np.full([2, 1000], 0.5)
MeanError = []
"""




class CMACNetwork:
    def __init__(self):
        self.weights = np.full([2, ], 0.5)
        self.d = [[2, 3, 1], [3, 2, 1]]
        self.d5 = []
        self.MeanError = []
        self.y1 = []
        self.y2 = []
        self.x1 = []
        self.x2 = []

    def read_samples():
        csv_file = open('samples.csv','r')
        for y1, y2, x1, x2 in csv.reader(csv_file, delimiter=','):
            # Append each variable to a separate list
            self.y1.append(float(y1))
            self.y2.append(float(y2))
            self.x1.append(float(x1))
            self.x2.append(float(x2))

    
    def table_index(self, y):
        # y is two scalar input
        p = np.empty([RECEPTIVE_FIELD_SIZE, DIM_INPUT])
        miu = np.empty([3])
        for i in range(0, RECEPTIVE_FIELD_SIZE):
            for j in range(0, DIM_INPUT):
                p[i][j] = round((y[j] + self.d[j][i]) / RECEPTIVE_FIELD_SIZE)   #d
            miu[i] = self.hash(p[i])
        return miu

    def mapping(self, y):
        # output x_predict
        miu = self.table_index(y)  # miu1,2,3

        x = np.empty([2])
        for i in range(0, 2):
            x[i] = 0
            for j in range(0, RECEPTIVE_FIELD_SIZE):
                x[i] += self.weights[i][int(miu[j])]  # where input the weight
        return x

    def train_step(self, y, x_pred, x_true):
        miu = self.table_index(y)  # miu1,2,3 figure out where is the weight
        error = x_true - x_pred
        for i in range(0, DIM_OUTPUT):
            increment = LEARNING_RATE * error[i] / RECEPTIVE_FIELD_SIZE
            # print("error:", error[0])
            # print("increment:", increment)
            for j in range(0, RECEPTIVE_FIELD_SIZE):
                self.weights[i][int(miu[j])] += increment

        meanSError = (error[0] ** 2 + error[1] ** 2) ** 0.5
        print("Error:", meanSError)
        return meanSError

    def readfile(self, data_address):
        df = pd.read_csv(data_address)
        y1 = list(df['saveX'])
        y2 = list(df['saveY'])
        x1 = list(df['savePitch'])
        x2 = list(df['saveRoll'])
        y = [y1, y2]
        x = [x1, x2]
        return y, x

    def quantization(self, y):
        # input raw y, output quantized y, list
        min_y = min(y)
        max_y = max(y)
        Result = []
        for one_input in y:
            result = round(RESOLUTION_RECEPTIVE * (one_input - min_y) / (max_y - min_y))
            if result >= RESOLUTION_RECEPTIVE:
                result =  RESOLUTION_RECEPTIVE - 1
            Result.append(result)
        return Result

    def train(self, y_raw, x):
        # y is after quantization
        # y_raw, x = self.readfile("samples.csv")
        # print(y_raw[0])
        num = len(y_raw[0])
        y1 = self.quantization(y_raw[0])
        y2 = self.quantization(y_raw[1])
        y = [y1, y2]
        print()
        print(y)
        for epoch in range(0, TRAINING_ITERATIONS):
            print("--------------------epoch", epoch, "------------------------")
            error = 0
            for i in range(0, num):
                y_pair = [y[0][i], y[1][i]]
                x_pair = [x[0][i], x[1][i]]
                x_pred = self.mapping(y_pair)
                if (epoch == 999): plt.scatter(x=x_pred[0], y=x_pred[1], c="red")
                error += self.train_step(y_pair, x_pred, x_pair)
                print("input:", y_pair, "pred:", x_pred, "true:", x_pair)
            MeanError.append(error)


if __name__ == '__main__':
    df = pd.read_csv('samples.csv')
    y1 = list(df['saveX'])
    y2 = list(df['saveY'])
    x1 = list(df['savePitch'])
    x2 = list(df['saveRoll'])
    y = [y1, y2]
    x = [x1, x2]
    cmac = CMACNetwork()
    cmac.train(y, x)

        
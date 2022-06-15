import numpy as np
import matplotlib.pyplot as plt
import csv
DISPLACEMENT_n5 = [3, 0, 2, 4, 1]
DISPLACEMENT_n3 = [1, 0, 2]

RESOLUTION_RECEPTIVE = 50
RECEPTIVE_FIELD_SIZE = 5 # n_a
DIM_INPUT = 2
DIM_OUTPUT = 2
LEARNING_RATE = 0.01
TRAINING_ITERATIONS = 1000
TRAINING_SAMPLES = 150

# 315 4 233 4
# -0.4862360954284668 -1.1059720516204834 0.3141592741012573 -0.3896780014038086
weights = np.full([2, 50, 50], 0)


class CMACNetwork:
    def __init__(self):
        self.weights = np.zeros([2, 50, 50])
        self.weights_index = np.zeros([50, 50])
        self.y1_min = 4
        self.y1_max = 315
        self.y2_min = 4
        self.y2_max = 233

        #self.x1_min = -2.0857
        #self.x1_max = 2.0857
        #self.x2_min = -1.3265
        #self.x2_max = 0.3142

        self.x1_min = -1.2
        self.x1_max = -0.5
        self.x2_min = -0.4
        self.x2_max = 0.32

        self.MeanError = 0
        self.step_size = 3  # displacement
        self.y1 = []
        self.y2 = []
        self.x1 = []
        self.x2 = []

        
    def read_samples(self):
        csv_file = open('/home/bio/Desktop/BIHR_Nao2022/src/tutorial_3/scripts/samples.csv','r')
        for y1, y2, x1, x2 in csv.reader(csv_file, delimiter=','):
            # Append each variable to a separate list
            #print("-----")
            #print(y1)
            self.y1.append(float(y1))
            self.y2.append(float(y2))
            self.x1.append(float(x1))
            self.x2.append(float(x2))

        #print("---------------")
        #print(self.y1)

    def quantization(self, x1, x2, y1, y2):
        x1_q = int((x1 - self.x1_min) * RESOLUTION_RECEPTIVE/(self.x1_max - self.x1_min))
        x2_q = int((x2 - self.x2_min) * RESOLUTION_RECEPTIVE / (self.x2_max - self.x2_min))
        y1_q = int((y1 - self.y1_min) * RESOLUTION_RECEPTIVE / (self.y1_max - self.y1_min))
        y2_q = int((y2 - self.y2_min) * RESOLUTION_RECEPTIVE / (self.y2_max - self.y2_min))
        if (x1_q >= RESOLUTION_RECEPTIVE): x1_q = 49
        if (x2_q >= RESOLUTION_RECEPTIVE): x2_q = 49
        if (y1_q >= RESOLUTION_RECEPTIVE): y1_q = 49
        if (y2_q >= RESOLUTION_RECEPTIVE): y2_q = 49

        if (x1_q < 0): x1_q = 0
        if (x2_q < 0): x2_q = 0
        if (y1_q < 0): y1_q = 0
        if (y2_q < 0): y2_q = 0

        return x1_q, x2_q, y1_q, y2_q

    def de_quantization(self, x1_q, x2_q):
        x1 = x1_q *(self.x1_max - self.x1_min)/RESOLUTION_RECEPTIVE + self.x1_min
        x2 = x2_q * (self.x2_max - self.x2_min) / RESOLUTION_RECEPTIVE + self.x2_min
        # y1 = y1_q * (self.y1_max - self.y1_min) / RESOLUTION_RECEPTIVE + self.y1_min
        # y2 = y2_q * (self.y2_max - self.y2_min) / RESOLUTION_RECEPTIVE + self.y2_min
        return x1, x2

    def weightindex_init(self):
        for i in range(0, RESOLUTION_RECEPTIVE):
            for j in range(0, RESOLUTION_RECEPTIVE):
                if (i + RESOLUTION_RECEPTIVE*j) % self.step_size == 0:
                    # print("index:", i, j)
                    self.weights_index[i][j] = 1

    def localization_2(self, y1, y2):
        shift = int((RECEPTIVE_FIELD_SIZE - 1) / 2)
        # print("y12:", y1, y2)
        y1_lower_bound = y1 - shift
        y1_upper_bound = y1 + shift
        y2_lower_bound = y2 - shift
        y2_upper_bound = y2 + shift
        # print("bound:", y1_lower_bound, y1_upper_bound, y2_lower_bound, y2_upper_bound)
        if y1 < shift:  # shift = 1 if r=3; shift = 2 if r = 5
            y1_lower_bound = 0
        if y1 > RESOLUTION_RECEPTIVE - 1 - shift:  # 48 if r =3; 47 if r=5
            y1_upper_bound = RESOLUTION_RECEPTIVE - 1
        if y2 < shift:  # shift = 1 if r=3; shift = 2 if r = 5
            y2_lower_bound = 0
        if y2 > RESOLUTION_RECEPTIVE - 1 - shift:  # 48 if r =3; 47 if r=5
            y2_upper_bound = RESOLUTION_RECEPTIVE - 1

        # print("bound:", y1_lower_bound, y1_upper_bound, y2_lower_bound, y2_upper_bound)

        loc = []
        for i in range(y1_lower_bound, y1_upper_bound+1):
            for j in range(y2_lower_bound, y2_upper_bound+1):
                if self.weights_index[i][j] == 1:
                    loc.append([i, j])
        # print("loc:", loc)
        return loc

    def train_step(self, y1, y2, x1, x2):
        loc = self.localization_2(y1, y2)
        x_true = [x1, x2]
        x_pred = self.mapping(y1, y2)
        Error = []
        for i in range(0, DIM_OUTPUT):
            error = x_true[i] - x_pred[i]
            Error.append(error**2)
            # print('error:', error)
            for j in range(0, len(loc)):
                self.weights[i][loc[j][0]][loc[j][1]] += LEARNING_RATE * error / RECEPTIVE_FIELD_SIZE
                # print(LEARNING_RATE * error / RECEPTIVE_FIELD_SIZE)
                # weights[i][int(loc[j][0])][int(loc[j][1])] += LEARNING_RATE * error / RECEPTIVE_FIELD_SIZE
        return (Error[0] + Error[1])**0.5

    def mapping(self, y1, y2):
        loc = self.localization_2(y1, y2)
        x_pred = [0, 0]
        if len(loc)!=0:
            for i in range(0, DIM_OUTPUT):
                for j in range(0, len(loc)):
                    # j is index of location, i is index of row or column
                    # print(i, int(loc[j][0]), int(loc[j][1]))
                    x_pred[i] += self.weights[i][loc[j][0]][loc[j][1]]
                x_pred[i] = int(x_pred[i])
                # x_pred[i] += weights[i][int(loc[j][0])][int(loc[j][1])]
        # print(x_pred)
        return x_pred

    def execute(self):
        self.read_samples()
        self.weightindex_init()
        #print(self.x1)

        for epoch in range(0, TRAINING_ITERATIONS):
            error = 0
            for i in range(0, TRAINING_SAMPLES):
                x1_q, x2_q, y1_q, y2_q = self.quantization(self.x1[i], self.x2[i], self.y1[i], self.y2[i])
                # print("xy", x1_q, x2_q, y1_q, y2_q)
                error += self.train_step(y1_q, y2_q, x1_q, x2_q)
                self.mapping(y1_q, y2_q)
            print('error:', error)
        print(self.weights)


'''
if __name__ == '__main__':

    # weights = np.full([2, 50, 50], 0)
    RECEPTIVE_FIELD_SIZE = 3
    cmac = CMACNetwork()
    cmac.read_samples()
    MSE = []
    cmac.weightindex_init()
    # ax = sns.heatmap(cmac.weights_index, cmap="YlGnBu") # visual the displacement
    # plt.show()
    # cmac.localization_2(49, 49)
    # x = cmac.de_quantization(22,33,12,54)
    # print(x)
    for epoch in range(0, TRAINING_ITERATIONS):
        MeanError = 0
        for i in range(0, TRAINING_SAMPLES):
            x1_q, x2_q, y1_q, y2_q = cmac.quantization(cmac.x1[i], cmac.x2[i], cmac.y1[i], cmac.y2[i])
            # print("xy", x1_q, x2_q, y1_q, y2_q)
            MeanError += cmac.train_step(y1_q, y2_q, x1_q, x2_q)
            cmac.mapping(y1_q, y2_q)
        print("epoch", epoch, "MSE", MeanError/TRAINING_SAMPLES)
        MSE.append(MeanError/len(cmac.y1))
    Weight = cmac.weights[0]
    #plt.plot(MSE, 'blue', label="caseB - r3")
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import matplotlib

class cmac:
    def __init__(self):
        self.resolution = 50
        self.n_a = 3  # receptive field size
        self.n_x = 2
        self.n_y = 2
        # n_w = 1000
        self.miu = []
        self.lr = 0.01

        self.df = pd.read_csv('samples.csv')
        self.d = list(np.empty([2, n_a]))
        # d = [[]]
        # displacement
        self.d[0][0] = 2
        self.d[0][1] = 3
        self.d[0][2] = 1
        self.d[1][0] = 3
        self.d[1][1] = 2
        self.d[1][2] = 1

        # W = np.empty([2, 500])
        self.W = np.full([2, 1000], 0.5)
        self.y1 = list(df['saveX'])
        self.y2 = list(df['saveY'])
        self.x1 = list(df['savePitch'])
        self.x2 = list(df['saveRoll'])

    def normalize_y1(x):
        result = round(self.resolution*(x - min(self.y1))/(max(self.y1)-min(self.y1)))
        if result >= self.resolution:
            return result - 1
        else:
            return result


    def normalize_y2(x):
        result = round(self.resolution*(x - min(self.y2))/(max(self.y2)-min(self.y2)))
        if result >= self.resolution:
            return result - 1
        else:
            return result


    def normalize_x1(x):
        result = round(self.resolution*(x - min(self.x1))/(max(self.x1)-min(self.x1)))
        if result >= self.resolution:
            return result - 1
        else:
            return result


    def normalize_x2(x):
        result = round(self.resolution*(x - min(self.x2))/(max(self.x2)-min(self.x2)))
        if result >= self.resolution:
            return result - 1
        else:
            return result


    def cmac_hash(p):
        self.r_k = round((self.resolution - 2)/self.n_a) + 2
        h = 0
        # print("r_k:", r_k, "h:", h)
        for i in range(0, self.n_y):
            h = h*r_k + p[i]
        return int(h)


    def table_index(y):
        # y is two scalar input
        p = np.empty([n_a, n_y])
        self.miu = np.empty([3])
        for i in range(0, n_a):
            for j in range(0, n_y):
                p[i][j] = round((y[j] + d[j][i])/n_a)
            self.miu[i] = cmac_hash(p[i])
        return miu


    def cmac_mapping(y):
        # output x_predict
        self.miu = table_index(y)   # miu1,2,3

        x = np.empty([2])
        for i in range(0, 2):
            x[i] = 0
            for j in range(0, n_a):
                x[i] += self.W[i][int(miu[j])]
        return x


    def cmac_train(y, x_pred, x_true):
        self.miu = table_index(y)  # miu1,2,3 figure out where is the weight
        self.error = x_true - x_pred
        for i in range(0, n_x):
            increment = lr*error[i]/n_a
            # print("error:", error[0])
            # print("increment:", increment)
            for j in range(0, n_a):
                self.W[i][int(miu[j])] += increment

        meanSError = (error[0]**2 + error[1]**2)**0.5
        print("Error:", meanSError)
        return meanSError


    def cmac_generate_weights(self):
        y1_norm = list(map(normalize_y1, y1))
        y2_norm = list(map(normalize_y2, y2))
        x1_norm = list(map(normalize_x1, x1))
        x2_norm = list(map(normalize_x2, x2))
        y = [y1_norm, y2_norm]
        x = [x1, x2]
        plt.scatter(x=x1, y=x2, c="blue")
        #plt.show()
        MeanError = []
        for epoch in range(0, 1000):
            print("--------------------epoch", epoch, "------------------------")
            error = 0
            for i in range(0, len(y1_norm)):
                y_pair = [y[0][i], y[1][i]]
                x_pair = [x[0][i], x[1][i]]
                x_pred = cmac_mapping(y_pair)
                if(epoch==999): plt.scatter(x=x_pred[0], y=x_pred[1], c="red")
                error += cmac_train(y_pair, x_pred, x_pair)
                print("input:", y_pair, "pred:", x_pred, "true:", x_pair)
            MeanError.append(error)
        plt.savefig("Scatter.png")
        plt.plot(MeanError)
        plt.savefig("MAE.png")
        # plt.show()

    # receives the input from the blob detector 
    def cmac_map_input_output(self, y1, y2):
        y = empty([2])
        x = empty([2])
        y[0] = y1
        y[1] = y2
        pitch = 0
        roll = 0
        x = self.cmac_mapping(y)

        # TODO: map normalized output to joint range:
        # joint ranges: pitch: -2.0857; 2.0857
        # joint ranges: roll: -1.3265; 0.3142

        pitch = x[0]
        roll = x[1]
        return pitch, roll

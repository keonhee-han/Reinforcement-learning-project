import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
DISPLACEMENT_n5 = [3, 0, 2, 4, 1]
DISPLACEMENT_n3 = [1, 0, 2]

RESOLUTION_RECEPTIVE = 50
RECEPTIVE_FIELD_SIZE = 3 # n_a
DIM_INPUT = 2
DIM_OUTPUT = 2
LEARNING_RATE = 0.01
TRAINING_ITERATIONS = 1000

# 315 4 233 4
# -0.4862360954284668 -1.1059720516204834 0.3141592741012573 -0.3896780014038086
weights = np.full([2, 50, 50], 0)

class CMACNetwork:
    def __init__(self):
        self.weights = np.zeros([2, 50, 50])
        self.y1_min = 4
        self.y1_max = 315
        self.y2_min = 4
        self.y2_max = 233
        self.x1_min = -1.2
        self.x1_max = -0.5
        self.x2_min = -0.4
        self.x2_max = 0.32
        self.MeanError = 0

    def quantization(self, x1, x2, y1, y2):
        x1_q = int((x1 - self.x1_min) * RESOLUTION_RECEPTIVE/(self.x1_max - self.x1_min))
        x2_q = int((x2 - self.x2_min) * RESOLUTION_RECEPTIVE / (self.x2_max - self.x2_min))
        y1_q = int((y1 - self.y1_min) * RESOLUTION_RECEPTIVE / (self.y1_max - self.y1_min))
        y2_q = int((y2 - self.y2_min) * RESOLUTION_RECEPTIVE / (self.y2_max - self.y2_min))
        if (x1_q >= RESOLUTION_RECEPTIVE): x1_q = 49
        if (x2_q >= RESOLUTION_RECEPTIVE): x2_q = 49
        if (y1_q >= RESOLUTION_RECEPTIVE): y1_q = 49
        if (y2_q >= RESOLUTION_RECEPTIVE): y2_q = 49
        return x1_q, x2_q, y1_q, y2_q

    def localization(self, y1, y2, displacement):
        # y1, y2 are indexs
        # output weight location on the map

        location = []
        if (y1 > RESOLUTION_RECEPTIVE -1 - int((RECEPTIVE_FIELD_SIZE - 1)/2)):
            start_y1 = RESOLUTION_RECEPTIVE - RECEPTIVE_FIELD_SIZE
        elif (y1 < int((RECEPTIVE_FIELD_SIZE - 1)/2)):
            start_y1 = 0
        else:
            start_y1 = y1 - (RECEPTIVE_FIELD_SIZE-1)/2

        if (y2 > RESOLUTION_RECEPTIVE -1- int((RECEPTIVE_FIELD_SIZE - 1)/2)):
            start_y2 = RESOLUTION_RECEPTIVE - RECEPTIVE_FIELD_SIZE
        elif (y1 < int((RECEPTIVE_FIELD_SIZE - 1)/2)):
            start_y2 = 0
        else:
            start_y2 = int(y2 - (RECEPTIVE_FIELD_SIZE-1)/2)
        #print("start:", start_y1, start_y2)

        for i in range(0, RECEPTIVE_FIELD_SIZE): # 0,1,2
            location.append([start_y1 + i, start_y2 + displacement[i]])
        return location

    def train_step(self, y1, y2, x1, x2, displacement):
        loc = self.localization(y1, y2, displacement)
        x_true = [x1, x2]
        x_pred = self.mapping(y1, y2, displacement)
        Error = []
        for i in range(0, DIM_OUTPUT):
            error = x_true[i] - x_pred[i]
            Error.append(error**2)
            # print('error:', error)
            for j in range(0, RECEPTIVE_FIELD_SIZE):
                self.weights[i][int(loc[j][0])][int(loc[j][1])] += LEARNING_RATE * error / RECEPTIVE_FIELD_SIZE
                # print(LEARNING_RATE * error / RECEPTIVE_FIELD_SIZE)
                # weights[i][int(loc[j][0])][int(loc[j][1])] += LEARNING_RATE * error / RECEPTIVE_FIELD_SIZE
        return (Error[0] + Error[1])**0.5

    def mapping(self, y1, y2, displacement):
        # get the predict value
        loc = self.localization(y1, y2, displacement)
        x_pred = [0, 0]
        for i in range(0, DIM_OUTPUT):
            for j in range(0, RECEPTIVE_FIELD_SIZE):
                # print(i, int(loc[j][0]), int(loc[j][1]))
                x_pred[i] += self.weights[i][int(loc[j][0])][int(loc[j][1])]
                # x_pred[i] += weights[i][int(loc[j][0])][int(loc[j][1])]
        # print(x_pred)
        return x_pred


if __name__ == '__main__':
    df = pd.read_csv('samples.csv')
    y1 = list(df['saveX'])
    y2 = list(df['saveY'])
    x1 = list(df['savePitch'])
    x2 = list(df['saveRoll'])
    # weights = np.full([2, 50, 50], 0)
    cmac = CMACNetwork()
    # cmac.localization(y1, y2)
    # MeanError = 0
    MSE = []
    for epoch in range(0, TRAINING_ITERATIONS):
        MeanError = 0
        for i in range(0, len(y1)):
            x1_q, x2_q, y1_q, y2_q = cmac.quantization(x1[i], x2[i], y1[i], y2[i])
            # print("xy", x1_q, x2_q, y1_q, y2_q)
            MeanError += cmac.train_step(y1_q, y2_q, x1_q, x2_q, DISPLACEMENT_n3)
            cmac.mapping(y1_q, y2_q, DISPLACEMENT_n3)
        print("epoch", epoch, "MSE", MeanError/len(y1))
        MSE.append(MeanError/len(y1))
    Weight = cmac.weights[0]
    print("Weight:", Weight)
    #ax = sns.heatmap(Weight, cmap="YlGnBu")
    #plt.savefig("heatmap_3.png")
    # plt.show()
    # plt.plot(MSE, 'blue', label="caseB-r3")

    # ---------------- 75 training samples -------------
    y1 = list(df['saveX'][0:75])
    y2 = list(df['saveY'][0:75])
    x1 = list(df['savePitch'][0:75])
    x2 = list(df['saveRoll'][0:75])
    MSE = []
    cmac = CMACNetwork()
    for epoch in range(0, TRAINING_ITERATIONS):
        MeanError = 0
        for i in range(0, len(y1)):
            x1_q, x2_q, y1_q, y2_q = cmac.quantization(x1[i], x2[i], y1[i], y2[i])
            # print("xy", x1_q, x2_q, y1_q, y2_q)
            MeanError += cmac.train_step(y1_q, y2_q, x1_q, x2_q, DISPLACEMENT_n3)
            cmac.mapping(y1_q, y2_q, DISPLACEMENT_n3)
        print("epoch", epoch, "MSE", MeanError/len(y1))
        MSE.append(MeanError/len(y1))
    #plt.plot(MSE, 'red', label="caseA")
    #plt.legend(loc = "upper right")
    #plt.title("MSE over Epochs")
    #plt.savefig('MSEoverEpoch.png')
    #plt.show()

    # ---------------- 150 training samples n-a 5-------------
    RECEPTIVE_FIELD_SIZE = 5
    y1 = list(df['saveX'][0:75])
    y2 = list(df['saveY'][0:75])
    x1 = list(df['savePitch'][0:75])
    x2 = list(df['saveRoll'][0:75])
    MSE = []
    cmac = CMACNetwork()
    for epoch in range(0, TRAINING_ITERATIONS):
        MeanError = 0

        for i in range(0, len(y1)):
            x1_q, x2_q, y1_q, y2_q = cmac.quantization(x1[i], x2[i], y1[i], y2[i])
            # print("xy", x1_q, x2_q, y1_q, y2_q)
            MeanError += cmac.train_step(y1_q, y2_q, x1_q, x2_q, DISPLACEMENT_n5)
            cmac.mapping(y1_q, y2_q, DISPLACEMENT_n5)
        print("epoch", epoch, "MSE", MeanError / len(y1))
        MSE.append(MeanError / len(y1))
    Weight = cmac.weights[0]
    print("Weight:", Weight)
    ax = sns.heatmap(Weight, cmap="YlGnBu")
    plt.savefig("heatmap_5.png")
    #plt.plot(MSE, 'red', label="caseB-r5")
    #plt.legend(loc="upper right")
    #plt.title("MSE over Epochs")
    #plt.savefig('MSEoverEpoch-2.png')
    #plt.show()


    # print(cmac.weights[0])



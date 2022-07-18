from sklearn import tree
import numpy as np
import copy

Q_suppose = np.zeros((10, 3))



class RL_DT:
    def __init__(self, init_state=5, gamma=0.9, MAXSTEPS=300):
        # self.A = {'Left': 0, 'Right': 1, 'Kick': 2}  # Possible action
        self.A = [0, 1, 2]  # 'Left': 0, 'Right': 1, 'Kick': 2
        self.sM = []  # set of all state
        self.visit = np.zeros((10, 3))  # counting the amount of visited state
        self.visit_state = np.zeros(10)

        self.Q = np.zeros((10, 3))  # q table
        self.reward_true = np.zeros((10, 3))
        self.transitionTree = tree.DecisionTreeClassifier()
        self.rewardTree = tree.DecisionTreeClassifier()
        # self.Pm = np.zeros((10,3))
        self.Rm = np.zeros((10, 3)) # reward matrix
        self.Ch = False
        self.exp = False
        self.gamma = gamma
        self.state_num = 10
        self.maxstep = MAXSTEPS
        self.init_state = init_state
        self.possible_state = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        self.X_train = []
        self.y_train = []


    def reward_init(self):
        for i in range(10):
            for j in range(3):
                if j == 0 or j == 1:
                    self.reward_true[i][j] = -1
                if j == 2:
                    self.reward_true[i][j] = -2
                if j == 2 and (i == 3 or i == 5):
                    self.reward_true[i][j] = 20
                if j == 2 and (i == 9 or i == 0):
                    self.reward_true[i][j] = -20
        print(self.reward_true)


    def State_Transition(self, state, action):
        shift = 0
        if action == 0:
            shift = -1
        elif action == 1:
            shift = 1
        next_state = state + shift
        if next_state < 0 or next_state > 9:
            return state
        return next_state


    def get_predictions(self, s_m, a_m):
        r_pred = self.rewardTree.predict([[s_m, a_m]])
        return r_pred[0]


    def add_experience_to_tree(self, s, action, r):
        self.X_train.append([s, action])
        self.y_train.append(r)
        self.rewardTree.fit(self.X_train, self.y_train)
        return True


    def Update_Model(self,s,action,r, s_prime):
        # not completed
        n = self.state_num
        self.Ch = self.add_experience_to_tree(s, action, r)
        for s_m in self.sM:
            for a_m in self.A:
                # print("pred:", s_m, a_m, self.get_predictions(s_m, a_m))
                self.Rm[s_m][a_m] = self.get_predictions(s_m, a_m)
        return self.Ch
                # self.Rm[s_m][a_m] = self.reward_true[s_m][a_m]


    def Check_Model(self):
        for r in np.nditer(self.Rm):
            if r > 0:
                return True
        return False



    def Compute_Value_test(self, stepsize):
        # Value iteration
        # print("Compute_value")
        minivisits = np.min(self.visit)
        print("visit:", self.visit)
        for step in range(0, stepsize):
            for s in self.sM:
                for a in self.A:
                    if self.exp and self.visit[s][a] == minivisits:
                        # print("RMax")
                        self.Q[s][a] = 999
                    else:
                        # print("R")
                        self.Q[s][a] = self.Rm[s][a]
                        s_prime = self.State_Transition(s, a)
                        self.Q[s][a] += self.gamma*max(self.Q[s_prime][:])

        return 0

    def q_max(self, state):
        Q = self.Q[state][:]
        max_q = Q[0]
        max_i = 0
        for i in range(len(Q)):
            if Q[i] > max_q:
                max_q = Q[i]
                max_i = i
        # print("max_action:", max_i)
        return max_i


    def execute(self):
        s = self.init_state
        self.sM.append(s)
        # print(self.Q[s][:])
        # self.visit_dict_state[s] = 0
        for step in range(self.maxstep):
            action = self.q_max(s)  # greedy action
            print("maxaction:", action)
            self.visit[s][action] += 1
            s_prime = self.State_Transition(s, action)
            # print("visit:", self.visit_dict)
            r = self.reward_true[s][action]
            if s_prime not in self.sM:
                self.sM.append(s_prime)
            self.Update_Model(s, action, r, s_prime)
            # self.exp = self.Check_Model()
            self.exp = True
            # print("exp:", self.exp)
            if step>100:
                self.exp = False
            if self.Ch:
                # self.Compute_Value(300)
                self.Compute_Value_test(500)
            s = s_prime
            print(self.Q)
        print(self.sM)
        print(self.Rm)

        return 0


if __name__ == '__main__':
    RL_DT = RL_DT()
    RL_DT.reward_init()
    RL_DT.execute()

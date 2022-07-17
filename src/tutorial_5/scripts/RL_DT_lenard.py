from sklearn import tree
import numpy as np
import copy


def RobotMotionLookUP(state, action):
    shift = 0
    if (action == 0):
        shift = -1
    elif (action == 1):
        shift = 1

    next_state = state + shift
    if (next_state < 0 or next_state > 9):
        return state
    return next_state


class RL_DT:
    def __init__(self, current_state=5, Rmax=100, gamma=0.9, MAXSTEPS=100):
        self.A = {'Left': 0, 'Right': 1, 'Kick': 2}  # Possible action
        self.sM = np.zeros((10))  # set of all state
        self.visit_number = np.zeros((10, 3))  # counting the amount of visited state
        self.Q = np.zeros((10, 3))  # q table
        self.Q[0][0] = -10000  # Punish going out of boundaries
        self.Q[9][1] = -10000
        self.current_state = current_state  # Current state of robot leg
        self.next_state = current_state
        self.Rmax = Rmax  # For exploration mode, giving least visited state some reward for making explotation
        self.transitionTree = tree.DecisionTreeClassifier()
        self.rewardTree = tree.DecisionTreeClassifier()
        # self.Pm = np.zeros((10,3))
        self.Rm = np.zeros((10, 3))
        self.inputTree = np.zeros(2)
        self.Ch = False
        self.exp = False
        self.deltaTransition = 0
        self.deltaReward = 0
        self.gamma = gamma
        self.MAXSTEPS = MAXSTEPS
        self.previous_state_action = []
        self.previous_action = []
        self.previous_reward = []
        

    def add_experience_reward(self,state,action,reward):
        self.previous_state_action.append([state,action])
        self.previous_action.append(reward)
        x = self.previous_state_action
        y = self.previous_action
        self.rewardTree = self.rewardTree.fit(x, y)  # must be of form samples,
        return True


    def get_predictions(self, state, action):
        return self.rewardTree.predict([[action, state]])

    def update_model(self,state, action, reward):
        self.add_experience_reward(state,action,reward)
        for state in range(len(self.sM)):
            for action in range(len(self.A)):
                 prediction = self.get_predictions(state, action)
                 self.Rm[state][action] = prediction

        return True

    def check_model(self):
        self.exp = np.all(self.Rm[:, :] < 0)

    def check_convergence(self, action_values_temp):
        for i in range(self.Q.shape[0]):
            for j in range(self.Q.shape[1]):
                if (abs(self.Q[i][j] - action_values_temp[i][j]) > 0.01):
                    return False
        return True

    def compute_values(self):

        minvisit = np.min(self.visit_number)
        converged = False
        while not converged:
            action_values_temp = copy.deepcopy(self.Q)
            for s in range(0, self.Q.shape[0]):
                for a in range(0, self.Q.shape[1]):
                    if (s == 0 and a == 0 or s == 9 and a == 1):
                        continue
                    if self.exp and self.visit_number[s][a] == minvisit:
                        self.Q[s][a] = self.Rmax

                    else:
                        self.Q[s][a] = self.Rm[s, a]
                    s_next = RobotMotionLookUP(s, a)

                    self.Q[s][a] += self.gamma * np.max(self.Q[s_next][:])
            converged = self.check_convergence(action_values_temp)

    def execute_action(self, action):
        self.next_state = RobotMotionLookUP(self.current_state, action)
        # please update reward function
        print("Your current state: " + str(self.current_state))
        print("Next state: " + str(self.next_state))

        # Hardcoded rewards for testing purposes
        reward = -1
        #if(action==2 and (self.current_state == 0 or self.current_state == 1)):
        #    reward = 20
        return reward


    def execute(self):
        for i in range(1000):

            self.exp = True
            action = np.argmax(self.Q[self.current_state])
            print("Choosing action: " + str(action))
            reward = self.execute_action(action)
            print(reward)
            self.visit_number[self.current_state][action] += 1
            print("Current visit-table: ")
            print(self.visit_number)
            self.update_model(self.current_state, action, reward)
            #self.Rm[self.current_state][action] = reward
            print("Reward Tree: ")
            print(self.Rm)
            self.Ch = True

            if self.Ch:
                self.compute_values()

            print("Q-Table: ")
            print(self.Q)
            self.current_state = self.next_state




if __name__ == '__main__':
    current_location = 5
    RL_DT = RL_DT(current_location)
    RL_DT.execute()

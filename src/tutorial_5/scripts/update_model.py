from sklearn import tree
import numpy as np

# Test class for the update algorithm 
class simulation:

    def __init__(self):
        self.stateSet = np.zeros(10)  # states already visited/known
        self.reward = {'Goal': 20, 'Move_Leg': -1, 'Fail': -2, 'Fall': -20} # just exampe values
        #self.possibleStates = [0,1,2,3,4,5,6,7,8,9]
        self.possibleActions = {'Left': 0, 'Right': 1, 'Kick': 2}
        self.transitionTree = tree.DecisionTreeClassifier()
        self.rewardTree = tree.DecisionTreeClassifier()
        self.Pm = np.zeros((10,3))
        self.Rm = np.zeros((10,3))
        self.inputTree = np.zeros(2)
        self.deltaTransition = 0
        self.deltaReward = 0
    # input parameters:
    # prev_state: discretized value of the Hip
    # action: action determined by the Q-table (either 0, 1 or 2)
    # reward: scalar value, given by us based on the result of the action
    # next_state: state reaced after the action execution
    # output:
    # state, if model hs changed -> currently always true, must be changed
    # Reward and Transition probability predictions of shape (10,3)
    def update_model(self, prev_state, action,reward, next_state):
        rel_state_change = prev_state - next_state # should result in -1, 0 or 1
        changed = self.add_experience_trans(prev_state, action, rel_state_change)
        self.add_experience_reward(reward)
        for state in range(len(self.stateSet)):
            for action in range(len(self.possibleActions)):
                self.Pm[state, action] = self.combine_results(state, action)
                self.Rm[state, action] = self.get_predictions(state, action)
        print(self.Pm)
        print(self.Rm)
        return True, self.Pm, self.Rm

    # Update the tranisition tree based on the relative change of the state
    def add_experience_trans(self, state, action, state_change):
        tmp = np.append(np.array(action), np.array(state))
        self.inputTree = np.vstack((self.inputTree, tmp))
        self.deltaTransition = np.append(self.deltaTransition, state_change)
        self.transitionTree = self.transitionTree.fit(self.inputTree, self.deltaTransition)
        return True

    def add_experience_reward(self,  reward):
        self.deltaReward = np.append(self.deltaReward, reward)
        self.rewardTree = self.rewardTree.fit(self.inputTree, self.deltaReward) # must be of form samples, 
        return True

    # Compute probabilities of the state change
    def combine_results(self, state, action):
        prob = np.max(self.transitionTree.predict_proba([[action,state]]))
        return prob

    def get_predictions(self, state, action):
        return self.rewardTree.predict([[action, state]])
        

if __name__=='__main__':
    # test with arbitrary values
    state = np.zeros(1)
    action = 1       
    reward = -1
    state_ = np.array([1])
    test = simulation()
    test.update_model(state, action, reward, state_)
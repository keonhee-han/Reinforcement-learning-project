from sklearn import tree
import numpy as np


STATES = 10
EPISODES = 200


class algorithm:

    def __init__(self, hip_states):
        # Total number of states
        self.Ns_leg = hip_states
        self.Sm = np.zeros(STATES)
        self.state = 9

        # Define actions
       
        self.possibleActions = [0, 1, 2]
        
        # Visit count
        self.visits = np.zeros((self.Ns_leg, len(self.possibleActions)))
        
        # Prob transitions
        self.Pm = np.zeros((self.Ns_leg, len(self.possibleActions)))
        self.Rm = np.zeros((self.Ns_leg, len(self.possibleActions)))
        
        # Initialize Decision Trees
        self.transitionTree = tree.DecisionTreeClassifier()
        self.rewardTree = tree.DecisionTreeClassifier()

        # Initialize input (always same) and output vectors for trees
        self.x_array = np.zeros(2)
        self.deltaS = np.array((0))
        self.deltaR = np.array((0))

        # Define rewards
        self.goal_reward = 20 # Reward for scoring goal
        self.miss_penalty = -2 # Miss the goal
        self.fall_penalty = -20 # Penalty for falling over
        self.action_penalty = -1 # Penalty for each action execution
        
        # Learning parameters
        self.gamma = 0.001 # Discount factor
        
        # Q values for state and action
        self.Q = np.zeros((self.Ns_leg, len(self.possibleActions)))

        # Rewards for simulation
        self.stateRewards = [[-20, -1, -20],
                             [-1, -1, -2],
                             [-1, -1, -2],
                             [-1, -1, -2],
                             [-1, -1, -2],
                             [-1, -1, -2],
                             [-1, -1, 20],
                             [-1, -1, -2],
                             [-1, -1, -2],
                             [-1, -20, -20]]

    def allowed_actions(self, s1):
        # Generate list of actions allowed depending on nao leg state
        actions_allowed = []
        if (s1 < self.Ns_leg - 2):  # No passing furthest left kick
            actions_allowed.append(self.action_dict["left"])
        if (s1 > 1):  # No passing furthest right kick
            actions_allowed.append(self.action_dict["right"])
        actions_allowed.append(self.action_dict["kick"]) # always able to kick
        actions_allowed = np.array(actions_allowed, dtype=int)
        return actions_allowed
    
    def get_reward(self, state, action):
        reward = self.stateRewards[state][action]
        return reward
                 
    def get_action(self, state):
        print("State passed for Q Table: ", state)
        actionIndex = np.argmax(self.Q[state])

        # make sure that we do not go out of bounds
        if state == 0 and actionIndex == 0:
            if self.Q[state, 1] > self.Q[state, 2]:
                actionIndex = 1
            else:
                actionIndex = 2
        elif state == 9 and actionIndex == 1:
            if self.Q[state, 0] > self.Q[state, 2]:
                actionIndex = 0
            else:
                actionIndex = 2
        print("Action Index determined by Q table: ", actionIndex)
        return actionIndex
    
    def check_model(self, state):
        exp = np.all(self.Rm[int(state), :] < 0)
        return exp
    
    def add_experience(self, n, state, action, delta):
        if n == 0:
            # x (input)
            x = np.append(np.array(action), state)
            self.x_array = np.vstack((self.x_array, x))

            # y (output)
            self.deltaS = np.append(self.deltaS, delta)
            self.transitionTree = self.transitionTree.fit(self.x_array, self.deltaS)
       
        elif n == 2:
            self.deltaR = np.append(self.deltaR, delta)
            self.rewardTree = self.rewardTree.fit(self.x_array, self.deltaR)
        CH = True
        return CH
    
    def combine_results(self, sm1, am):
        deltaS1_prob = np.max(self.transitionTree.predict_proba([[am, sm1]]))
        P_deltaS = deltaS1_prob 


        return P_deltaS
    
    def get_predictions(self, sm1, am):
        deltaR_pred = self.rewardTree.predict([[am, sm1]])
        return deltaR_pred
    
    def update_model(self, state, action, reward, state_):
        n = 1
        CH = False
        xi = np.zeros(n)
        for i in range(n):
            xi[i] = state - state_
            CH = self.add_experience(i, state, action, xi[i])

        CH = self.add_experience(n+1, state, action, reward)

        for sm in range(len(self.Sm)):
            for am in range(len(self.possibleActions)):
                self.Pm[sm, am] = self.combine_results(sm, am)
                self.Rm[sm, am] = self.get_predictions(sm, am)
        return CH
    
    def compute_values(self, exp):
        minvisits = np.min(self.visits)
        for sm in range(len(self.Sm)):
            
            for am in range(len(self.possibleActions)):
                if exp and self.visits[sm,am] == minvisits:
                    self.Q[sm,am] = self.goal_reward
                else:
                    self.Q[sm, am] = self.Rm[sm, am]
                    for sm1_ in range(self.Ns_leg):
                        
                        self.Q[sm, am] += self.gamma * self.Pm[sm1_,  am] \
                        * np.max(self.Q[sm1_, :])
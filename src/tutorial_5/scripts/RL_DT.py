from sklearn import tree
import numpy as np



def RobotMotionLookUP(state,action):
    # the transition of state, action Q?
    next_state_LU =state
    if (state=="0" and action=="2"):
        next_state_LU = 0
    if (state=="1" and action=="2"):
        next_state_LU  = 1
    if (state=="2" and action=="2"):
        next_state_LU  = 2
    if (state=="3" and action=="2"):
        next_state_LU  = 3
    if (state=="4" and action=="2"):
        next_state_LU  = 4
    if (state=="5" and action=="2"):
        next_state_LU  = 5
    if (state=="6" and action=="2"):
        next_state_LU  = 6
    if (state=="7" and action=="2"):
        next_state_LU  = 7
    if (state=="8" and action=="2"):
        next_state_LU  = 8
    if (state=="9" and action=="2"):
        next_state_LU  = 9
    if (state=="0" and action=="1"):
        next_state_LU  = 1
    if (state=="1" and action=="1"):
        next_state_LU  = 2
    if (state=="2" and action=="1"):
        next_state_LU  = 3
    if (state=="3" and action=="1"):
        next_state_LU  = 4
    if (state=="4" and action=="1"):
        next_state_LU  = 5
    if (state=="5" and action=="1"):
        next_state_LU  = 6
    if (state=="6" and action=="1"):
        next_state_LU  = 7
    if (state=="7" and action=="1"):
        next_state_LU  = 8
    if (state=="8" and action=="1"):
        next_state_LU  = 9
    if (state=="9" and action=="1"):
        next_state_LU  = 9
    if (state=="0" and action=="0"):
        next_state_LU  = 0
    if (state=="1" and action=="0"):
        next_state_LU  = 0
    if (state=="2" and action=="0"):
        next_state_LU  = 1
    if (state=="3" and action=="0"):
        next_state_LU  = 2
    if (state=="4" and action=="0"):
        next_state_LU  = 3
    if (state=="5" and action=="0"):
        next_state_LU  = 4
    if (state=="6" and action=="0"):
        next_state_LU  = 5
    if (state=="7" and action=="0"):
        next_state_LU  = 6
    if (state=="8" and action=="0"):
        next_state_LU  = 7
    if (state=="9" and action=="0"):
        next_state_LU  = 8
    
    return next_state_LU 





class RL_DT:
    def __init__(self,current_state =5,Rmax = 5,gamma = 0.9,MAXSTEPS =100):
        self.A = {'Left': 0, 'Right': 1, 'Kick': 2} #Possible action
        self.sM = np.zeros((10)) # set of all state
        self.visit_number = np.zeros((10,3)) # counting the amount of visited state
        self.Q =np.zeros((10,3)) # q table 
        self.current_state = current_state # Current state of robot leg
        self.next_state = current_state
        self.Rmax = Rmax # For exploration mode, giving least visited state some reward for making explotation
        self.transitionTree = tree.DecisionTreeClassifier()
        self.rewardTree = tree.DecisionTreeClassifier()
        #self.Pm = np.zeros((10,3))
        self.Rm = np.zeros((10,3))
        self.inputTree = np.zeros(2)
        self.Ch = False
        self.exp = False
        self.deltaTransition = 0
        self.deltaReward = 0
        self.gamma = gamma
        self.MAXSTEPS = MAXSTEPS
        self.action = np.zeros((1,1))

    def add_experience_trans(self, state, action, state_change):
        tmp = np.append(np.array(action), np.array(state))
        self.inputTree = np.vstack((self.inputTree, tmp))
        self.deltaTransition = np.append(self.deltaTransition, state_change)
        #self.transitionTree = self.transitionTree.fit(self.inputTree, self.deltaTransition)
        return True

    def add_experience_reward(self,  reward):
        self.deltaReward = np.append(self.deltaReward, reward)
        self.rewardTree = self.rewardTree.fit(self.inputTree, self.deltaReward) # must be of form samples, 
        return True

    def combine_results(self, state, action):
        prob = np.max(self.transitionTree.predict_proba([[action,state]]))
        return prob
        

    def get_predictions(self, state, action):
        return self.rewardTree.predict([[action, state]])


    def update_model(self,action,reward):
        rel_state_change = self.current_state - self.next_state # should result in -1, 0 or 1
        self.Ch = self.add_experience_trans(self.current_state, action, rel_state_change)
        self.add_experience_reward(reward)
        for state in range(len(self.sM)):
            for action in range(len(self.A)):
                #self.Pm[state, action] = self.combine_results(state, action)
                self.Rm[state, action] = self.get_predictions(state, action)
        #print(self.Pm)
        #print(self.Rm)
        return True

    def check_model(self):
        self.exp = np.all(self.Rm[:, :] < 0) # what does it mean? the exp

    def compute_values(self):
        K = np.zeros((10,1))
        for s in range(0,self.visit_number.shape[0]):
            for a in range(0,3):
                if self.visit_number[s][a] > 0:
                    K[s]  = 0
                else:
                    K[s] = 9999999
        minvisit = np.min(self.visit_number)
        for s in range(0,self.Q.shape[0]):
            for a in range(0,self.Q.shape[1]):
                # check all state-action value in Q?
                if self.exp and self.visit_number[s][a] == minvisit:
                    self.Q[s][a] = self.Rmax
                elif K[s] > self.MAXSTEPS:
                    self.Q[s][a] = self.Rmax
                else:
                    self.Q[s][a] = self.Rm[s,a]

                s_next = RobotMotionLookUP(s,a)
                if K[s]+1 < K[s_next]:
                    K[s_next] = k[s] +1
                self.Q[s][a] = self.gamma*1*np.max(self.Q[s_next][:]) # value iteration?


    def execute_action(self,action):
        self.next_state = RobotMotionLookUP(self.current_state,action)
        #please update reward function
        reward = np.random.choice([0,1])  # so the reward is random
        return reward
       #reward = ()
       #Please update reward there




    def execute(self):
        while True: 
            # 1. Implement code to find argmax of state s
            arr = self.Q[self.current_state][:]
            loc = arr == np.max(arr)
            action = np.zeros((1,1))
            if loc[0] == True:
                action=0
            elif loc[1] == True:
                action=1
            elif loc[2] == True:
                action=2
            print(action)
            # 2. Execute a and receive rward and observe next state s'
            reward = self.execute_action(action)
            # 3. incremenet visits(s,a)
            self.visit_number[self.current_state][action] = self.visit_number[self.current_state][action] +1 
            # 4. Update model so (self.pm,self.rm and self.ch), as input just action and reward used because
            # other variable defined in class as public, so we have access and there is no need to return because 
            # they are also public
            self.Ch = self.update_model(action,reward)
            # 5. model check whether it is exporation mode or not
            # no input again global varialbe

            self.check_model()
            # 6. if CH is true update q table us
            if self.Ch:
                self.compute_values()
            # 7. Update current state
            self.current_state = self.next_state
            print(self.Q)
            #print("running beaches")



# Usage of code
if __name__=='__main__':
    current_location = 0
    how_less_greedy_algorithm_for_unknown = 5
    RL_DT = RL_DT(current_location,how_less_greedy_algorithm_for_unknown)
    RL_DT.execute()
        
    

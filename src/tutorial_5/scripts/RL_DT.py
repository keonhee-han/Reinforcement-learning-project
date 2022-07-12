from sklearn import tree
import numpy as np

class RL_DT:
    def __init__(self,current_state =5,Rmax = 5):
        self.A = {'Left': 0, 'Right': 1, 'Kick': 2} #Possible action
        self.sM = np.zeros((10)) # set of all state
        self.visit_number = np.zeros((10,3)) # counting the amount of visited state
        self.Q =np.zeros((10,3)) # q table 
        self.current_state = current_state # Current state of robot leg
        self.next_state = current_state
        self.Rmax = Rmax # For exploration mode, giving least visited state some reward for making explotation
        self.Pm 
        self.Rm 
        self.Ch
        self.exp

    def update_model(self,action,reward):

    def check_model(self):
    #Algorithm one loop part(Algorithm-1 RL-DT from line 5 to 19)
    def execute(self):
        while : 
            # 1. Implement code to find argmax of state s
            arr = Q[self.current_state][:]
            action = npwhere(arr == np.amax(arr))
            # 2. Execute a and receive rward and observe next state s'
            reward,self.next_state = execute_action(action)
            # 3. incremenet visits(s,a)
            self.visit_number[self.current_state][action] = self.visit_number[self.current_state][action] +1 
            # 4. Update model so (self.pm,self.rm and self.ch), as input just action and reward used because
            # other variable defined in class as public, so we have access and there is no need to return because 
            # they are also public
            self.update_model(action,reward)
            # 5. model check whether it is exporation mode or not
            # no input again global varialbe
            self.check_model()
            # 6. if CH is true update q table us
            if self.Ch:
                self.compute_values()
            # 7. Update current state
            self.current_state = self.next_state 



# Usage of code
if __name__=='__main__':
    current_location = 5
    how_less_greedy_algorithm_for_unknown = 5
    RL_DT = RL_DT(current_location,how_less_greedy_algorithm_for_unknown)
    RL_DT.execute()
        
    
from sklearn import tree
import numpy as np

class RL_DT_test:  # execute action at given state (as discretized state indices)
    def __init__(self, RMax_, s_):
        states = range(10)  # number of states (assume discretized leg distance from the hip is 10
        self.train_epsiodes = 100   # random guess
        self.gamma = 0.9 # discount_factor
        self.exp_ = False
        self.initial_state = initial_states
        ##[hkh] initialize each visit for each state zero for each state
        self.actions = {"move_right": 0, "move_left": 1, "kick": 2}
        self.reward = {"move_leg": -1, "fall": -20, "fail_goal": -2, "goal": 20 }
        # self.G_t = 0 # total future reward up to given time t (for one episode)
        # self.stateSet = {s: self.actions for s in states}  # =visits: state-action pair is initially zero, no states visited so far
        self.Q_table = np.zeros((len(states), len(self.actions)))   # reward table
        # self.Q_table = np.array([ [-20, -1, -1], [-1, -1, -1] , [-1, -1, -1] , [-1, -1, -1]
        #                           , [-1, -1, -1] , [-1, -1, -1] , [-1, -1, -1] , [-1, -1, -1]
        #                           , [-1, -1, -1] , [-1, -1, -20] ])
        self.visit_table = np.zeros((len(states), len(self.actions)))
        self.stateSet = np.array(range(states))

        self.leg_state_abs = 0 #-0.379472 to 0.790477 -> discretized by 10 ~ 0.117 per bin
        self.leg_state_dis = 10   # 0 - 9, 10 for invalid

        ##[hkh] we have to implement DT for transition probability. DT
        # (reminder: in DT example, the state should be vaild within the possible range.
        # e.g. In A=L node, if True for x=0 meaning no movement, then its output is either 0:no movement as true, -1:moved left
        # In A=R, if x=1: moved right as true, x=0: idle. its output 0 as True or Y=1 as False.

    def train(self):
        s_next = 0
        for _ in range(self.train_epsiodes):  # end if s <- s'
            a_ = self.opt_policy(s_)  # [hkh] its Utility func is determined by reward and transition func that are determined by DT
            if a_ == 2: self.kick()
            elif a_ == 1: self.move_in()
            elif a_ == 0: self.move_out()
            else: print("Nothing is given for optimal policy!") # 2. execute action a -> move_left, move_right or kick
            # After taking an action, monitoring the state of the robot so that we can reward for that state-action pair.
            reward_type = self.state_monitor(input_wait)  # check what happened to robot after taken the action and give'em reward
            # 3. Upon taking an action, receives reward, observe next state
            R_ += self.reward[reward_type] - self.reward["move_leg"]  # current reward: According to the algorithm, punish with amount -2 as it's moved.
            # self.Q_table[s_, a_] += R_ + self.gamma * self.Q_table[s_, a_]
            self.visit_table[s_, a_] += 1  # increase the state-action visits counter
            if a_ == "move_left":
                s_next = s_ - 1
            elif a_ == "move_right":
                s_next = s_ + 1
            # 4. reaches a new state s' <=> observe new state -> just read in leg angle again
            # s_new, _, CH_ = model_.update_model(s_, a_)
            # 5. check if new state has already been visited <=> check if it's in S_M
            # if not s_new in self.S_M:   # if not, add it to the stateSet in add_visited_states
            #     self.S_M.add(s_new)
            ## [hkh] new state action initializaiton is already done in constructor.
            # 6. Update model -> many substeps
            Alg2 = update_model(state=s_, action=a_, reward=R_, next_state=s_next, S_M=self.visit_table,A_=self.actions)
            P_M, R_M, CH_ = Alg2()
            # 7. Check policy, if exploration/exploitation mode
            exp_ = self.check_model(P_M, R_M)
            # 8. Compute values -> many substeps
            if CH_:
                Alg3 = compute_values(RMax_, P_M, R_M, self.visit_table, exp_)
                Alg3()
            # If kick happened, end an action sequence as one episode
            if a_ == "kick": break
            state = s_next
            return state

    def opt_policy(self, state):
        action_next = np.argmax(self.Q_table[state, :])
        return action_next

    def discretize_leg(self):
        self.leg_state_dis = np.round((self.leg_state_abs - LEG_MIN) / (LEG_MAX - LEG_MIN) * 9)

    def state_monitor(self):  # This is where the keyboard should input to give reward during certain time.
        print("move_leg:1, fall:2, fail_goal:3, goal: 4")
        value = input("Please input the reward type:\n")
        if value==1:reward_type="move_leg"
        elif value==2:reward_type="fall"
        elif value==3:reward_type="fail_goal"
        elif value==4:reward_type="goal"
        print(f'You entered {reward_type}')
        return reward_type


    def opt_policy(self, state, next_action):  # optimal policy functio that chooses the action maximizing the reward


    def transition_func(self, state, action):
        return new_state


    # 1.
    def get_action_policy(self):
        pass


    # 3.
    def get_reward(self):  # Reward function: R(s,a)

    # maybe with keyboard instead of tactile buttons -> 4 types of reward


    # 6.
    def add_visited_states(self):
        for state in self.stateSet:
            if state == self.leg_state_dis:
                return
        self.stateSet.append(self.leg_state_dis)


    def check_model(self, P_M, R_M):  # Check Policy c.f. 1st paper
        if np.sum(R_M) < 0.4:
            return True
        elif R_M > 0.4:
            return False
        else:
            print("check model: R_M error!")


# Test class for the update algorithm
class update_model:
    def __init__(self, state, action, reward, next_state, S_M, A_):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.S_M = S_M  # stateSet
        self.reward = reward
        self.A_ = A_
        self.transitionTree = tree.DecisionTreeClassifier()
        self.rewardTree = tree.DecisionTreeClassifier()
        self.inputTree = np.zeros(2)
        self.deltaTransition = 0
        self.deltaReward = 0

        self.P_M = np.asarray(S_M)
        self.R_M = np.asarray(S_M)
    '''
    input parameters:
    state_: current state discretized value of the distance from hip to the foot
    action: action determined by the Q-table (among 0, 1 or 2)
    reward: scalar value, given by us based on the result of the action
    next_state: state reached after the action execution
    output:
    CH_: state, if model hs changed -> currently always true, must be changed
    Reward and Transition probability predictions of shape (10,3)
    '''
    def __call__(self, *args, **kwargs):
        CH_ = False
        relative_state_change = self.next_state - self.state  # should result in -1, 0 or 1
        CH_ = self.add_experience_trans(self.state, self.action, relative_state_change)
        # Update the tree for reward
        CH_ = self.add_experience_reward(self.reward)    # update each tree incrementally with input vector and desired output
        ## Tree updating is done.
        # Combine results for each tree into model
        for s_M in self.S_M:
            for a_M in range(len(self.A_)):
                self.P_M[s_M, a_M] = self.combine_results(s_M, a_M)
                self.R_M[s_M, a_M] = self.get_predictions(s_M, a_M)
        print(self.P_M)
        print(self.R_M)
        return True, self.P_M, self.R_M, CH_


    # Update the tranisition tree based on the relative change of the state
    def add_experience_trans(self, state, action, relative_state_change):
        tmp = np.append(np.array(action), np.array(state))  # input vector
        self.inputTree = np.vstack((self.inputTree, tmp))
        self.deltaTransition = np.append(self.deltaTransition, relative_state_change)
        self.transitionTree = self.transitionTree.fit(self.inputTree, self.deltaTransition)
        return True

    def add_experience_reward(self, reward):
        self.deltaReward = np.append(self.deltaReward, reward)
        self.rewardTree = self.rewardTree.fit(self.inputTree, self.deltaReward)  # must be of form samples,
        return True

    # Compute probabilities of the state change
    def combine_results(self, state, action):
        prob = np.max(self.transitionTree.predict_prob([[action, state]]))
        return prob

    def get_predictions(self, state, action):
        return self.rewardTree.predict([[action, state]])


class compute_values:
    def __init__(self, RMax=20, P_M, R_M, S_M, A_, exp_, Gamma = 0.9, MAX_STEP = 10, Q_table):
        self.RMax = RMax
        self.P_M = P_M
        self.R_M = R_M
        self.visit_table = S_M  # stateSet
        self.A_ = A_
        self.exp_ = exp_
        self.gamma = Gamma
        self.K_ = np.asarray(S_M[:,0])
        self.min_visits = 0
        self.converged = False
        self.Q_table = Q_table
        self.MAX_STEP = MAX_STEP
        self.min_visits = 0

    def __call__(self, *args, **kwargs):
        # Initialize all state's step counts
        # self.steps_nearest_visited_state = {x: sys.maxint for x in range(9)}
        # visits_values = []
        # for state in range(len(self.visit_table)):
        #     for action in range(3):
                # if self.visit_table(state, action) > 0:
                #     self.K_(state) = 0
                # else:
                #     self.K_(state) = None
        self.min_visits = np.min(self.visit_table)
        while not self.converged:
            for state in range(len(self.visit_table)):
                for action in range(3):
                    if self.exp_ and self.visit_table[state, action] == self.min_visits:
                        # unknown states are given exploration bonus
                        self.Q_table[state, action] = self.RMax
                    # elif self.K_(state) > self.MAX_STEP:
                        # states out of reach
                        # self.Q_table = self.RMax
                    else:
                        # update remaining state's action-values
                        self.Q_table = self.R_M[state, action]
                        state_next = self.P_M[state, action]
                    self.Q_table[state, action] += self.gamma * 1 * np.max(state_next, action_next)


if __name__ == '__main__':
    # test with arbitrary values
    # state = np.zeros(1)
    # action = 1
    # reward = -1
    # state_ = np.array([1])
    # test = Algorithm_2()
    # test.update_model(state, action, reward, state_)
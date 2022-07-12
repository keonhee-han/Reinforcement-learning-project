#!/usr/bin/env python
from logging import StringTemplateStyle
import random
import rospy
from std_msgs.msg import String, Header
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from naoqi import ALProxy
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import csv
import random
from naoqi import ALProxy
import sys
# from MDP import *
from update_model import *

LEG_MAX = 0.790477
LEG_MIN = -0.379472

class tutorial5_soccer:
    def __init__(self, initial_states):
        self.jointPub = 0
        states = range(10)  # number of states (assume discretized leg distance from the hip is 10
        terminal_state = {0, 9}  ##[hkh] Let assume at the end of leg movement will be end state ## depending on DT Terminal states are differed

        # Initial variables for RL-DT
        self.exp_ = False
        self.S_M = set()   # set type to store and check visited state per episode
        self.initial_state = initial_states
        ##[hkh] initialize each visit for each state zero for each state
        self.actions = {"move_right": 0, "move_left": 0, "kick": 0}
        # self.actions = ["move_right", "move_left", "kick"]
        self.stateSet = {s: self.actions for s in states}  # =visits: state-action pair is initially zero, no states visited so far
        self.G_t = 0 # total future reward up to given time t
        self.train_epsiodes = 100   # random guess
        self.reward = {"move_leg": -1, "fall": -20, "kick_fail": -2, "goal": 20 }

        self.leg_state_abs = 0 #-0.379472 to 0.790477 -> discretized by 10 ~ 0.117 per bin
        self.leg_state_dis = 10   # 0 - 9, 10 for invalid

        ##[hkh] we have to implement DT for transition probability. DT
        # (reminder: in DT example, the state should be vaild within the possible range.
        # e.g. In A=L node, if True for x=0 meaning no movement, then its output is either 0:no movement as true, -1:moved left
        # In A=R, if x=1: moved right as true, x=0: idle. its output 0 as True or Y=1 as False.

    def RL_DT(self, RMax_, s_):   # execute action at given state
        self.S_M.add(s_)        # adding state to set of states visited initially

        while True: # end if s <- s'
            # 1. get next action a from policy (?)
            a_ = opt_policy(s_, a_current) #[hkh] its Utility func is determined by reward and transition func that are determined by DT

            # 2. execute action a -> move_left, move_right or kick
            if a_ == "kick": self.kick() elif a_ == "move_left": self.move_in() elif a_ == "move_right": self.move_out()
            else: print("Nothing is given for optimal policy!")
            reward_type = self.state_monitor(monitoring_time = 10) # After taking an action, monitoring the state of the robot so that we can reward for that state-action pair.

            # 3. Upon taking an action, receives reward
            self.G_t += self.reward[reward_type] - self.reward["move_leg"]  # According to the algorithm, punish with amount -2 as it's moved.
            self.stateSet[s_][a_] += 1  # increase the state-action visits counter

            # 4. reaches a new state s' <=> observe new state -> just read in leg angle again
            s_new = transition_func(s_, a_)

            # 5. check if new state has already been visited <=> check if it's in S_M
            if not s_new in self.S_M:   # if not, add it to the stateSet in add_visited_states
                self.S_M.add(s_new)
                ## [hkh] new state action initializaiton is already done in constructor.

            # 6. Update model -> many substeps
            P_M, R_M, CH_ = Algorithm_2.update_model()    # taking function from Lennard part.

            # 7. Check policy, if exploration/exploitation mode
            exp_ = self.check_model

            # 8. Compute values -> many substeps
            if CH_:
                self.compute_values(RMax_, P_M, R_M, self.S_M, exp_)
        pass

    def discretize_leg(self):
        self.leg_state_dis = np.round((self.leg_state_abs - LEG_MIN) / (LEG_MAX - LEG_MIN) * 9)

    def state_monitor(self, monitoring_time = 10):
        return reward_type

    def opt_policy(self, state, next_action):   # optimal policy functio that chooses the action maximizing the reward

    def transition_func(self, state, action):
        return new_state

    # 1.
    def get_action_policy(self):
        pass

    # 3.
    def get_reward(self):   # Reward function: R(s,a)
        # maybe with keyboard instead of tactile buttons -> 4 types of reward


    # 6.
    def add_visited_states(self):
        for state in self.stateSet:
            if state == self.leg_state_dis:
                return
        self.stateSet.append(self.leg_state_dis)

    # 7. 
    def increase_visits(self):
        pass
    
    # 9.
    def check_policy(self):
        pass

    def compute_values(self):




    # Callback function for reading in the joint values
    def joints_cb(self, data):
        self.joint_names = data.name  # LHipRoll for move in or move out
        self.joint_angles = data.position
        self.joint_velocities = data.velocity
        # get leg state
        for idx, names in enumerate(self.joint_names):
            if names == "LHipRoll":
                self.leg_state_abs = self.joint_angles[idx]
                self.discretize_leg()
                return
    


    # Read in the goal position!
    # TODO: Aruco marker detection
    def image_cb(self,data):
        bridge_instance = CvBridge()

    def set_joint_angles(self, head_angle, topic):
        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(topic) # each joint has a specific name, look into the joint_state topic or google  # When I
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.08 # keep this low if you can
        #print(str(joint_angles_to_set))
        self.jointPub.publish(joint_angles_to_set)

    def set_joint_angles_fast(self, head_angle, topic):
        # fast motion!! careful
        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(topic) # each joint has a specific name, look into the joint_state topic or google  # When I
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.4 # keep this low if you can
        #print(str(joint_angles_to_set))
        self.jointPub.publish(joint_angles_to_set)

    def set_joint_angles_list(self, head_angle_list, topic_list):
        # set the init one stand mode, doing it by all list together
        if len(head_angle_list) == len(topic_list):
            for i in range(len(topic_list)):
                head_angle = head_angle_list[i]
                topic = topic_list[i]
                joint_angles_to_set = JointAnglesWithSpeed()
                joint_angles_to_set.joint_names.append(topic) # each joint has a specific name, look into the joint_state topic or google  # When I
                joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
                joint_angles_to_set.relative = False # if true you can increment positions
                joint_angles_to_set.speed = 0.1 # keep this low if you can
                #print(str(joint_angles_to_set))
                self.jointPub.publish(joint_angles_to_set)
                rospy.sleep(0.05)

    # Moves its left hip back and forward and then goes back into its initial position
    def kick(self):
        # i think

        self.set_stiffness(True)
        # Set ankle position to zero
        #self.set_joint_angles(0.0, "LAnklePitch")
        # Set hip roll position to zero
        #self.set_joint_angles(0.0,"LHipPitch")
        # Move foot back
        self.set_joint_angles(0.48, "LHipPitch")
        rospy.sleep(1.0)

        self.set_joint_angles_fast(-1.1, "LHipPitch")
        # Move the foot back after kick
        rospy.sleep(2.0)
        self.one_foot_stand()
        #self.set_joint_angles(0.352, "LHipPitch")

    def set_initial_stand(self):
        robotIP = '10.152.246.137'
        try:
            postureProxy = ALProxy('ALRobotPosture', robotIP, 9559)
        except Exception, e:
            print('could not create ALRobotPosture')
            print('Error was', e)
        postureProxy.goToPosture('Stand', 1.0)
        print(postureProxy.getPostureFamily())

    def one_foot_stand(self):
        # it is the init state ready for kicking
        # careful !!!!!! very easy to fall
        print('one foot mode')
        self.set_stiffness(True)
        # using the rostopic echo to get the desired joint states, not perfect
        # rostopic echo /joint_states

        # way1 easy to fall when kick
        position = [0.004559993743896484, 0.5141273736953735, 1.8330880403518677, 0.19937801361083984, -1.9574260711669922,
                    -1.5124820470809937, -0.8882279396057129, 0.32840001583099365, -0.13955211639404297, 0.31297802925109863,
                    -0.3911280632019043, 1.4679961204528809, -0.8943638801574707, -0.12114405632019043, -0.13955211639404297,
                    0.3697359561920166, 0.23772811889648438, -0.09232791513204575, 0.07980990409851074, -0.3282339572906494,
                    1.676703929901123, -0.45717406272888184, 1.1964781284332275, 0.18872404098510742, 0.36965203285217285, 0.397599995136261]
        '''
        # way 2
        
        position = [-0.015382051467895508, 0.5120565295219421, 1.8346221446990967, 0.1779019832611084, -1.937483787536621,
                    -1.5124820470809937, -1.0692400932312012, 0.32760000228881836, -0.11807608604431152, 0.31297802925109863,
                    -0.3911280632019043, 1.4664621353149414, -0.8943638801574707, -0.12114405632019043, -0.11807608604431152,
                    0.3697359561920166, 0.2530679702758789, -0.09232791513204575, -0.07665801048278809, -0.269942045211792,
                    1.6951122283935547, -0.4617760181427002, 1.1949440240859985, 0.2025299072265625, 0.3589141368865967, 0.39399999380111694]
        '''
        # backup init position
        # way3 [0.004559993743896484, 0.5141273736953735, 1.8330880403518677, 0.15335798263549805, -1.9129400253295898, -1.5032780170440674, -1.199629783630371, 0.32760000228881836, -0.22852396965026855, 0.4019498825073242, -0.3911280632019043, 1.4679961204528809, -0.8943638801574707, -0.12114405632019043, -0.22852396965026855, 0.3697359561920166, 0.23772811889648438, -0.09232791513204575, 0.08594608306884766, -0.3052239418029785, 1.6905097961425781, -0.44950389862060547, 1.1995460987091064, 0.1994619369506836, 0.3512439727783203, 0.3952000141143799]

        # way2 not ok [-0.015382051467895508, 0.5120565295219421, 1.8346221446990967, 0.1779019832611084, -1.937483787536621, -1.5124820470809937, -1.0692400932312012, 0.32760000228881836, -0.11807608604431152, 0.31297802925109863, -0.3911280632019043, 1.4664621353149414, -0.8943638801574707, -0.12114405632019043, -0.11807608604431152, 0.3697359561920166, 0.2530679702758789, -0.09232791513204575, -0.07665801048278809, -0.269942045211792, 1.6951122283935547, -0.4617760181427002, 1.1949440240859985, 0.2025299072265625, 0.3589141368865967, 0.39399999380111694]
        joints =['HeadYaw', 'HeadPitch', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw',
                'LHand', 'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll', 'RHipYawPitch',
                'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll', 'RShoulderPitch', 'RShoulderRoll',
                'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand']
        self.set_joint_angles_list(position, joints)

    def move_in(self):
        print('move in')
        LHipRoll_angle = self.joint_angles[self.joint_names.index('LHipRoll')]
        print(self.joint_names.index('LHipRoll'), LHipRoll_angle)
        if LHipRoll_angle > 0.45:
            increment = -0.03
            self.set_joint_angles(LHipRoll_angle + increment, "LHipRoll")
        else:
            print('low bound!!!')

    def move_out(self):
        print('move out')
        LHipRoll_angle = self.joint_angles[self.joint_names.index('LHipRoll')]
        #print(self.joint_names.index('LHipRoll'), LHipRoll_angle)
        if LHipRoll_angle < 0.74:
            increment = 0.03
            self.set_joint_angles(LHipRoll_angle + increment, "LHipRoll")
        else:
            print('upper bound')

    def set_initial_pos(self):
        # fix the joints to the initial positions for the standing position
        # I used the set_initial_stand instead - Tianle

        # Head
        self.set_stiffness(True)
        self.set_joint_angles(-0.13503408432006836, "HeadYaw")
        self.set_joint_angles(-0.49859189987182617, "HeadPitch")

        # Right arm
        self.set_joint_angles(-0.7439479827880859, "RShoulderPitch")
        self.set_joint_angles(-0.019984006881713867, "RShoulderRoll")
        self.set_joint_angles(0.30368995666503906, "RElbowYaw")
        self.set_joint_angles(0.49245595932006836, "RElbowRoll")
        self.set_joint_angles(1.2394300699234009, "RWristYaw")

        # Left arm
        self.set_joint_angles(1.8300200700759888, "LShoulderPitch")
        self.set_joint_angles(0.12421202659606934, "LShoulderRoll")
        self.set_joint_angles(-2.0856685638427734, "LElbowYaw")
        self.set_joint_angles(-0.08125996589660645, "LElbowRoll")
        self.set_joint_angles(-0.7424979209899902, "LWristYaw")

        # Left leg & foot
        self.set_joint_angles(-0.15642595291137695, "LHipYawPitch")
        self.set_joint_angles(0.2224719524383545, "LHipRoll")
        self.set_joint_angles(0.3528618812561035, "LHipPitch")
        self.set_joint_angles(0.1, "LKneePitch")
        self.set_joint_angles(-0.28996801376342773, "LAnklePitch")
        self.set_joint_angles(-0.24079608917236328, "LAnkleRoll")

        # Right leg & foot
        self.set_joint_angles(-0.15642595291137695, "RHipYawPitch")
        self.set_joint_angles(0.27616190910339355, "RHipRoll")
        self.set_joint_angles(0.3742539882659912, "RHipPitch")
        self.set_joint_angles(-0.09232791513204575, "RKneePitch")
        self.set_joint_angles(-0.11961007118225098, "RAnklePitch")
        self.set_joint_angles(-0.32050607681274414, "RAnkleRoll")       # That value is important for stability


    def set_stiffness(self, value):
        if value == True:
            service_name = '/body_stiffness/enable'
        elif value == False:
            service_name = '/body_stiffness/disable'
        try:
            stiffness_service = rospy.ServiceProxy(service_name, Empty)
            stiffness_service()
        except rospy.ServiceException, e:
            rospy.logerr(e)

    '''
    def touch_cb(self,data):
        if data.button == 1 and data.state == 1:  # TB1
            #self.set_stiffness(True)
            print("Kick motion & stiffness enabled")
            self.kick()
        if data.button == 2 and data.state == 1:
            self.set_stiffness(False)
            print("stiffness should be DISabled")
        if data.button == 3 and data.state == 1:
            self.set_stiffness(True)
            print("stiffness should be ENabled")
        # try kick motion
        #if data.button == 3 and data.state == 1:
        # left knee joint pitch: -0.092346 to 2.112528
        # Left hip joint pitch: -1.535889 to 0.484090
        # for RL motions the left hip roll is important: -0.379472 to 0.790477
    '''

    def touch_cb(self, data):
        if data.button == 1 and data.state == 1:  # TB1
            print("move in")
            self.move_in()
        if data.button == 2 and data.state == 1:
            print("move out")
            self.move_out()
        if data.button == 3 and data.state == 1:
            print("kick")
            self.kick()
        # try kick motion
        #if data.button == 3 and data.state == 1:
        # left knee joint pitch: -0.092346 to 2.112528
        # Left hip joint pitch: -1.535889 to 0.484090
        # for RL motions the left hip roll is important: -0.379472 to 0.790477

    def tutorial5_soccer_execute(self):

        # cmac training here!!!
        rospy.init_node('tutorial5_soccer_node', anonymous=True)
        self.set_stiffness(True)
        self.jointPub = rospy.Publisher("joint_angles", JointAnglesWithSpeed, queue_size=10)

        self.set_initial_stand()
        rospy.sleep(2.0)
        self.one_foot_stand()
        #rospy.Subscriber("joint_states",JointAnglesWithSpeed,self.joints_cb)
        rospy.Subscriber("tactile_touch", HeadTouch, self.touch_cb)
        rospy.Subscriber('joint_states', JointState, self.joints_cb)
        # start with setting the initial positions of head and right arm


        #rospy.Subscriber("/nao_robot/camera/top/camera/image_raw", Image, self.image_cb)

        rospy.spin()


class RL_DT(tutorial5_soccer):
    def __init__(self):
        # Set of actions
        self.actions = {"move_right": 0, "move_left": 1, "kick": 2}
        self.state_action = {s: 0 for s in self.states}   # state-action pair is initially zero, no states visited so far
        self.train_epsiodes = 100   # random guess
        self.reward = {"move_leg": -1, "fall": -20, "kick_fail": -2, "goal": 20 }


    def training(self):



if __name__=='__main__':
    ##[hkh]
    # MDP(init = init, actlist= action_list, terminals= terminal_state, states=states, transitions=transition_model)

    node_instance = tutorial5_soccer()
    #node_instance.one_foot_stand()
    node_instance.tutorial5_soccer_execute()
    #node_instance.stand()
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

LEG_MAX = 0.790477
LEG_MIN = -0.379472
RMax = 20
MAX_STEPS = 10 # Guessed value
DISCOUNT_FACTOR = 0.9
Convergence_Threshold = 0.1

class tutorial5_soccer:
    def __init__(self):
        self.jointPub = 0

        # Variables for RL-DT 
        self.actions = {"move_right": 0, "move_left": 1, "kick": 2}
        self.stateSet = []          # initially zero, no states visited so far
        self.train_episodes = 100   # random guess
        self.reward = {"move_leg": -1, "fall": -20, "kick_fail": -2, "goal": 20 }
        self.leg_state_abs = 0 #-0.379472 to 0.790477 -> discretized by 10 ~ 0.117 per bin
        self.leg_state_dis = 10   # 0 - 9, 10 for invalid
        self.visits = {} # Key: (state, action) - Value: visits
        self.steps_nearest_visited_state = {}
        self.exploration = False
        self.action_values = {} # Key: (state, action) - Value: value

    
    def discretize_leg(self):
        self.leg_state_dis = np.round((self.leg_state_abs - LEG_MIN) / (LEG_MAX - LEG_MIN) * 9)


    def training(self):
        
        # 1. get next action a from policy (?)
        # 2. execute action a -> move_left, move_right or kick
        # 3. get reward
        # 4. observe new state -> just read in leg angle again
        # 5. check if new state has already been visited
        # 6. if not, add it to the stateSet in add_visited_states
        # 7. increase the state-action visits counter
        # 8. Update model -> many substeps
        # 9. Check policy, if exploration/exploitation mode
        # 10. Compute values -> many substeps
        pass

    # 1.
    def get_action_policy(self):
        pass

    # 3.
    def get_reward(self):
        # maybe with keyboard instead of tactile buttons -> 4 types of reward
        pass

    # 6.
    def add_visited_states(self):
        for state in self.stateSet:
            if state == self.leg_state_dis:
                return
        self.stateSet.append(self.leg_state_dis)

    # 7. 
    def increase_visits(self):
        pass

    # 8.
    def update_model(self):
        pass
    
    # 9.
    def check_policy(self):
        pass

    def get_visits_state(self, state):
        visits_value = 0
        for action in self.actions:
            visits_value += self.visits.get((state, action), 0)
        return visits_value

    def get_reward_state_action_pair(self, state, action):
        # TODO implement
        return 0

    def get_next_states(self, state):
        # TODO: Implement
        return []

    def get_prop_next_state_given_state_action(self, next_state, current_state, action):
        # TODO: Implement
        return 0

    def get_next_state_action_value_greedy(self, state):
        return max([self.action_values[(state, action)] for action in self.actions])

    def check_convergence(self, action_values_temp):
        for state in self.stateSet:
            for action in self.actions:
                if self.action_values[(state, action)] != action_values_temp[(state, action)]:
                    return False
        return True

    def compute_values(self):

        # Initialize all state's step counts
        self.steps_nearest_visited_state = {x: sys.maxint for x in range(9)}
        visits_values = []
        for state in self.stateSet:
            visits_value = self.get_visits_state(state)
            visits_values.append(visits_value)
            if visits_value > 0:
                self.steps_nearest_visited_state[state] = 0
        min_visits = min(visits_values)

        # Perform value iteration on the model
        action_values_temp = {}
        converged = False
        while not converged:
            for state in self.stateSet:
                for action in self.actions:
                    if self.exploration and visits_value(state) == min_visits:
                        # Unknown states are given exploration bonus
                        action_values_temp[(state, action)] = RMax
                    elif self.steps_nearest_visited_state[state] > MAX_STEPS:
                        action_values_temp[(state, action)] = RMax
                    else:
                        # Update remaining state's action values
                        action_values_temp[(state, action)] = self.get_reward_state_action_pair(state, action)
                        for next_state in self.get_next_states(state):
                            if next_state not in self.stateSet:
                                self.stateSet.append(next_state)
                                for next_action in self.actions:
                                    self.visits[(next_state, next_action)] = 0
                            # Update action-values using Bellman Equation
                            action_values_temp[(state, action)] += \
                                DISCOUNT_FACTOR * \
                                self.get_prop_next_state_given_state_action(next_state, state, action) * \
                                self.get_next_state_action_value_greedy(next_state)
            converged = self.check_convergence(action_values_temp)
            self.action_values = action_values_temp





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


    def move_left(self):
        pass

    def move_right(self):
        pass

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
        except Exception e:
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


class RLDT:
    def __init__(self):
        # Set of actions
        self.actions = {"move_right": 0, "move_left": 1, "kick": 2}
        self.stateSet = []          # initially zero, no states visited so far
        self.train_epsiodes = 100   # random guess
        self.reward = {"move_leg": -1, "fall": -20, "kick_fail": -2, "goal": 20 }

    def training(self):





if __name__=='__main__':
    node_instance = tutorial5_soccer()
    #node_instance.one_foot_stand()
    node_instance.tutorial5_soccer_execute()
    #node_instance.stand()


#!/usr/bin/env python
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
from algorithm_module import algorithm

LEG_MAX = 0.790477
LEG_MIN = -0.379472

STATES = 10
EPISODES = 200

class tutorial5_soccer:
    def __init__(self):
        self.blobX = 0
        self.blobY = 0
        self.blobSize = 0
        self.shoulderRoll = 0
        self.shoulderPitch = 0
        # For setting the stiffnes of single joints
        self.jointPub = 0
        self.leg_state_dis = 10         # this variable must hold the current discrete state!!
        self.leg_state_abs = 0
        self.rldt = algorithm(STATES)

    # Callback function for reading in the joint values
    def joints_cb(self, data):
        #rospy.loginfo("joint states "+str(data.name)+str(data.position))
        # store current joint information in class variables
        self.joint_names = data.name  # LHipRoll for move in or move out
        self.joint_angles = data.position
        self.joint_velocities = data.velocity
        for idx, names in enumerate(self.joint_names):
            if names == "LHipRoll":
                self.leg_state_abs = self.joint_angles[idx]
                self.discretize_leg()
        
        #print("Current leg state: ", self.leg_state_dis)
        
    def discretize_leg(self):
        
        tmp = np.round((self.leg_state_abs - LEG_MIN) / (LEG_MAX - LEG_MIN) * 9.0)
       
        self.leg_state_dis = tmp


    # Read in the goal position!
    # TODO: Aruco marker detection
    def image_cb(self,data):
        bridge_instance = CvBridge()



    # TODO: put in cmac logic!
    # input to cmac mapping: self.blobX, self.blobY
    def move_arm(self):
        # for testing - random arm values

        self.set_joint_angles(self.shoulderPitch, "RShoulderPitch")
        self.set_joint_angles(self.shoulderRoll, "RShoulderRoll")



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
        joint_angles_to_set.speed = 0.2 # keep this low if you can
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
       
    # Currently, training is only done in simulation!! -> change position in which it will score a goal in the algorithm class
    def train(self):
        for episode in range(100):

            # Get action from optimal policy
            print("Current state: ", self.rldt.state)
            cur_state = self.rldt.state
            action = self.rldt.get_action(self.rldt.state)

            # Take action
            print("Next action: ", action)
           
            # Determine next state
            self.rldt.state = self.determine_state(self.rldt.state, action)
            print("New state: ", self.rldt.state)
            # Determine reward from action taken
            reward = self.rldt.get_reward(self.rldt.state, action)
           
            # Increment visits and update state set
            self.rldt.visits[int(cur_state), action] += 1         
            
            # Update model
            CH = self.rldt.update_model(cur_state, action, reward, self.rldt.state)
            exp = self.rldt.check_model(cur_state)

            if CH:
                self.rldt.compute_values(exp)
        print(self.rldt.Q)

    def execute(self):
        #self.rldt.state = 9           # arbitrary test state
        self.rldt.state = self.leg_state_dis
        print("State before finding kick position: ", self.rldt.state)
        # get actual state
        while True:
            action = self.rldt.get_action(self.rldt.state)
            print("Next action in execution: ", action)
            self.execute_action(action)
            if action == 2:
                print("kick state: ", self.rldt.state)
                break
            self.rldt.state = self.leg_state_dis          #self.determine_state(self.rldt.state, action)
            print("State: ", self.rldt.state)
        self.set_initial_stand()
        

        print("final state: ", self.rldt.state)



    def determine_state(self, state, action):
        if action == 0:
            state = state - 1
        elif action == 1:
            state = state +1
        elif action == 2:
            state = state
      
        return state


    def execute_action(self, actionIndex):
        if actionIndex == 0:    #Move In
            self.move_in()
        elif actionIndex == 1:
            self.move_out()
        elif actionIndex == 2:
            self.kick()
        rospy.sleep(1.0)







    def tutorial5_soccer_execute(self):
        rospy.init_node('tutorial5_soccer_node', anonymous=True)
        self.set_stiffness(True)
        self.jointPub = rospy.Publisher("joint_angles", JointAnglesWithSpeed, queue_size=10)
        self.set_initial_stand()
        rospy.sleep(2.0)
        self.one_foot_stand()
        #rospy.Subscriber("tactile_touch", HeadTouch, self.touch_cb)
        rospy.Subscriber('joint_states', JointState, self.joints_cb)
        #rospy.Subscriber("/nao_robot/camera/top/camera/image_raw", Image, self.image_cb)
        self.train()
        self.execute()
        rospy.spin()


if __name__=='__main__':
    node_instance = tutorial5_soccer()
    node_instance.tutorial5_soccer_execute()
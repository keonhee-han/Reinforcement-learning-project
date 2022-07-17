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


class tutorial5_soccer:
    def __init__(self):
        self.blobX = 0
        self.blobY = 0
        self.blobSize = 0
        self.shoulderRoll = 0
        self.shoulderPitch = 0
        # For setting the stiffnes of single joints
        self.jointPub = 0

    # Callback function for reading in the joint values
    def joints_cb(self, data):
        #rospy.loginfo("joint states "+str(data.name)+str(data.position))
        # store current joint information in class variables
        self.joint_names = data.name  # LHipRoll for move in or move out
        self.joint_angles = data.position
        self.joint_velocities = data.velocity

        pass

    # Read in the goal position!
    # TODO: Aruco marker detection
    def image_cb(self,data):
        bridge_instance = CvBridge()



    # TODO: put in cmac logic!
    # input to cmac mapping: self.blobX, self.blobY
    def move_arm(self):
        rospy.loginfo("---------------- here starts cmac movement -----------------------")
        # for testing - random arm values

        self.set_joint_angles(self.shoulderPitch, "RShoulderPitch")
        self.set_joint_angles(self.shoulderRoll, "RShoulderRoll")



    def set_joint_angles(self, head_angle, topic):
        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(topic) # each joint has a specific name, look into the joint_state topic or google  # When I
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.03 # keep this low if you can
        #print(str(joint_angles_to_set))
        self.jointPub.publish(joint_angles_to_set)

    def set_joint_angles_fast(self, head_angle, topic):
        # fast motion!! careful
        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(topic) # each joint has a specific name, look into the joint_state topic or google  # When I
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.6 # keep this low if you can
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
        # fast kick
        self.set_joint_angles_fast(-0.8, "LHipPitch")
        
        # Move the foot to original position
        rospy.sleep(2.0)
        #self.one_foot_stand()
        self.set_joint_angles(-0.3911280632019043, "LHipPitch")

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

        """
                position_ori = [0.004559993743896484, 0.5141273736953735, 1.8330880403518677, 0.19937801361083984, -1.9574260711669922,
                    -1.5124820470809937, -0.8882279396057129, 0.32840001583099365, -0.13955211639404297, 0.31297802925109863,
                    -0.3911280632019043, 1.4679961204528809, -0.8943638801574707, -0.12114405632019043, -0.13955211639404297,
                    0.3697359561920166, 0.23772811889648438, -0.09232791513204575, 0.07980990409851074, -0.3282339572906494,
                    1.676703929901123, -0.45717406272888184, 1.1964781284332275, 0.18872404098510742, 0.36965203285217285, 0.397599995136261]
        """

        # way1 the best position i find
        position = [0.004559993743896484, 0.5141273736953735, 1.8330880403518677, 0.19937801361083984, -1.9574260711669922,
                    -1.5124820470809937, -0.8882279396057129, 0.32840001583099365, -0.13955211639404297, 0.32,
                    -0.3911280632019043, 1.2, -0.4, -0.12114405632019043, -0.13955211639404297,
                    0.3697359561920166, 0.23772811889648438, -0.09232791513204575, 0.07980990409851074, -0.3282339572906494,
                    1.676703929901123, -0.8, 1.1964781284332275, 0.18872404098510742, 0.36965203285217285, 0.397599995136261]
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
        joints =['HeadYaw', 'HeadPitch', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw',
                 'LElbowRoll', 'LWristYaw','LHand', 'LHipYawPitch', 'LHipRoll',
                 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll', 'RHipYawPitch',
                 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll',
                 'RShoulderPitch', 'RShoulderRoll','RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand']
        self.set_joint_angles_list(position, joints)

    def move_in(self):
        print('move in')
        LHipRoll_angle = self.joint_angles[self.joint_names.index('LHipRoll')]
        print(self.joint_names.index('LHipRoll'), LHipRoll_angle)
        # original pos: 0.31297802925109863
        if LHipRoll_angle > 0.32:   # 0.45
            increment = -0.038  # -0.03
            self.set_joint_angles(LHipRoll_angle + increment, "LHipRoll")
            self.read_state_joint()
        else:
            print('low bound!!!')

    def move_out(self):
        print('move out')
        LHipRoll_angle = self.joint_angles[self.joint_names.index('LHipRoll')]
        #print(self.joint_names.index('LHipRoll'), LHipRoll_angle)
        if LHipRoll_angle < 0.7:   #0.74
            increment = 0.038
            self.set_joint_angles(LHipRoll_angle + increment, "LHipRoll")
            self.read_state_joint()
        else:
            print('upper bound')
            
    def read_state_joint:
        LHipRoll_angle = self.joint_angles[self.joint_names.index('LHipRoll')]
        state_joint = int((LHipRoll_angle - 0.32)/(0.7-0.32) * 10)
        print("state_joint")
        return state_joint
       
        
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

        # self.set_initial_stand()
        rospy.sleep(2.0)
        self.one_foot_stand()
        #rospy.Subscriber("joint_states",JointAnglesWithSpeed,self.joints_cb)
        rospy.Subscriber("tactile_touch", HeadTouch, self.touch_cb)
        rospy.Subscriber('joint_states', JointState, self.joints_cb)
        # start with setting the initial positions of head and right arm


        #rospy.Subscriber("/nao_robot/camera/top/camera/image_raw", Image, self.image_cb)

        rospy.spin()


if __name__=='__main__':
    node_instance = tutorial5_soccer()
    #node_instance.one_foot_stand()
    node_instance.tutorial5_soccer_execute()
    #node_instance.stand()


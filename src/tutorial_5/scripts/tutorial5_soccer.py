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
        # rospy.loginfo("joint states "+str(data.name)+str(data.position))
        # store current joint information in class variables
        self.joint_names = data.name
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
        joint_angles_to_set.speed = 0.1 # keep this low if you can
        #print(str(joint_angles_to_set))
        self.jointPub.publish(joint_angles_to_set)

    def set_joint_angles_list(self, head_angle_list, topic_list):
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

    # Moves its left hip back and forward and then goes back into its initial position
    def kick(self):
        self.set_stiffness(True)
        # Set ankle position to zero
        self.set_joint_angles(0.0, "LAnklePitch")
        # Set hip roll position to zero
        self.set_joint_angles(0.0,"LHipPitch")
        # Move foot back
        self.set_joint_angles(-0.1, "LHipPitch")
        rospy.sleep(1.0)

        self.set_joint_angles(-0.2, "LHipPitch")
        # Move the foot back after kick
        rospy.sleep(2.0)
        self.set_initial_pos()
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
        print('one foot mode')
        self.set_stiffness(True)
        #self.set_joint_angles(0.3, "LHipRoll")
        position = [-0.20099592208862305, -0.10128593444824219, 1.8530300855636597, 0.16716408729553223, -1.188891887664795,
               -0.03490658476948738, 0.6150920391082764, 0.2955999970436096, 0.052197933197021484, 0.17184996604919434,
               0.12122797966003418, 0.23466014862060547, -0.1764519214630127, -0.08432793617248535,
               0.052197933197021484, 0.07213997840881348, 0.2791459560394287, 0.2025299072265625, -0.3021559715270996,
               -0.09046411514282227, 1.7457342147827148, -0.19485998153686523, 1.2118180990219116, 0.03685808181762695,
               0.3573801517486572, 0.3255999684333801]
        joints =['HeadYaw', 'HeadPitch', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw',
                'LHand', 'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll', 'RHipYawPitch',
                'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll', 'RShoulderPitch', 'RShoulderRoll',
                'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand']
        self.set_joint_angles_list(position, joints)


    # fix the joints to the initial positions for the standing position
    def set_initial_pos(self):
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

        # start with setting the initial positions of head and right arm


        #rospy.Subscriber("/nao_robot/camera/top/camera/image_raw", Image, self.image_cb)

        rospy.spin()

if __name__=='__main__':
    node_instance = tutorial5_soccer()
    #node_instance.one_foot_stand()
    node_instance.tutorial5_soccer_execute()
    #node_instance.stand()


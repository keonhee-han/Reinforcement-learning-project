#!/usr/bin/env python
from random import randrange
import rospy
from std_msgs.msg import String, Header
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import csv
import random
import cmac

import HSV_Nao_blob_DET_v2 as NBD



class tutorial3_cmac:
    def __init__(self):
        self.cmac = cmac()
        self.blobX = 0
        self.blobY = 0
        self.blobSize = 0
        self.shoulderRoll = 0
        self.shoulderPitch = 0
        # For setting the stiffnes of single joints
        self.jointPub = 0


    # Read in the blob position
    def image_cb(self,data):
        bridge_instance = CvBridge()
        try:
            # br = CvBridge()
            # Output debugging information to the terminal
            rospy.loginfo("receiving video frame__B_DET")

            # Convert ROS Image message to OpenCV image
            src = bridge_instance.imgmsg_to_cv2(data,"bgr8") #rgb 8bit
            # current_frame = br.imgmsg_to_cv2(data)

            #[hkh]Getting x,y coordinate
            # cv_image = CVA.blob_detection(src)
            blob_detection_ret = NBD.blob_detection(src) #[hkh]
            if not isinstance(blob_detection_ret, type(None)):
                self.blobX, self.blobY = blob_detection_ret
            
            # cv2.imshow("Keypoints", cv_image) #Keypoints are already implemented above
            self.move_arm()
            cv2.waitKey(3)

        except CvBridgeError as e:
            rospy.logerr(e)


    # TODO: put in cmac logic!
    def move_arm(self):

        # for testing - random arm values
        rospy.loginfo("------ here insrt CMAC logic -----")
        test_pitch = random.uniform(-1, 1)
        test_roll = random.uniform(-1, 0.1)
        self.set_joint_angles(test_pitch, "RShoulderPitch")
        self.set_joint_angles(test_roll, "RShoulderRoll")


    def set_joint_angles(self, head_angle, topic):
        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(topic) # each joint has a specific name, look into the joint_state topic or google  # When I
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.1 # keep this low if you can
        print(str(joint_angles_to_set))
        self.jointPub.publish(joint_angles_to_set)



    # fix the joints to the initial positions
    def set_initial_pos(self):
        self.set_joint_angles("HeadYaw", -0.13503408432006836)
        self.set_joint_angles("HeadPitch", -0.49859189987182617)
        self.set_joint_angles("RShoulderPitch", -0.7439479827880859)
        self.set_joint_angles("RShoulderRoll", -0.019984006881713867)
        self.set_joint_angles("RElbowYaw", 0.30368995666503906)
        self.set_joint_angles("RElbowRoll",  0.49245595932006836)
        self.set_joint_angles("RWristYaw", 1.2394300699234009)
        



    def tutorial3_cmac_execute(self):
        self.cmac.cmac_generate_weights()
        rospy.init_node('tutorial3_cmac_node',anonymous=True) 
        rospy.Subscriber("joint_states",JointAnglesWithSpeed,self.joints_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10)

        # start with setting the initial positions of head and right arm
        self.set_initial_pos()

        rospy.spin()

if __name__=='__main__':
    node_instance = tutorial3_cmac()
    node_instance.tutorial3_cmac_execute()


#!/usr/bin/env python
import random
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
import ffnn_nao as nn

import HSV_Nao_blob_DET_v2 as NBD

### Authors:
### Keonhee Han
### Luisa Mayershofer
### Tianle Ni
### Batu Özen
### Lennard Riedel



class tutorial4_mlp:
    def __init__(self):
        self.blobX = 0
        self.blobY = 0
        self.blobSize = 0
        self.shoulderRoll = 0
        self.shoulderPitch = 0
        # For setting the stiffnes of single joints
        self.jointPub = 0
        # TDO: Initialize MLP here
        self.ffnn = nn.FFNN(sizes=[2, 64, 64, 2])


    # Read in the blob position
    def image_cb(self,data):
        bridge_instance = CvBridge()
        try:
            # Convert ROS Image message to OpenCV image
            src = bridge_instance.imgmsg_to_cv2(data,"bgr8") #rgb 8bit
            # current_frame = br.imgmsg_to_cv2(data)
            rospy.loginfo(len(src))
            
            blob_detection_ret = NBD.blob_detection(src) 
            if not isinstance(blob_detection_ret, type(None)):
                self.blobX, self.blobY = blob_detection_ret
                print('blobxy:', self.blobX, self.blobY )
                array = np.array([self.blobX/300.0, self.blobY/300.0])

                # Call prediction function from MLP NN to calculate the arm values based on the blob input
                self.shoulderPitch, self.shoulderRoll = self.ffnn.forward_prediction(array)
                self.move_arm()
            
            
            cv2.waitKey(3)

        except CvBridgeError as e:
            rospy.logerr(e)


    
    def move_arm(self):
        rospy.loginfo("---------------- here starts mlp movement -----------------------")
        
        self.set_joint_angles(self.shoulderPitch, "RShoulderPitch")
        self.set_joint_angles(self.shoulderRoll, "RShoulderRoll")


    def set_joint_angles(self, head_angle, topic):
        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(topic) # each joint has a specific name, look into the joint_state topic or google  # When I
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.1 # keep this low if you can
       
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

    def set_stiffness(self, value):
        if value == True:
            service_name = '/body_stiffness/enable'
        elif value == False:
            service_name = '/body_stiffness/disable'
        try:
            stiffness_service = rospy.ServiceProxy(service_name, Empty)
            stiffness_service()
        except rospy.ServiceException as e:
            rospy.logerr(e)
        



    def tutorial4_mlp_execute(self):

        rospy.init_node('tutorial4_mlp_node',anonymous=True)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10)
        # start with setting the initial positions of head and right arm
        self.set_stiffness(True)
        self.set_initial_pos()
        
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw", Image, self.image_cb)


        rospy.spin()

if __name__=='__main__':
    node_instance = tutorial4_mlp()
    node_instance.tutorial4_mlp_execute()


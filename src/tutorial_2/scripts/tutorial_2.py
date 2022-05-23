#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

import CV_Alg as CVA
import HSV_Nao_blob_DET_v2 as NBD

### Keonhee Han
### Luisa Mayershofer
### Batu Kaan Oezen
### Tianle Ni
### Lennard Riedel

class Central:


    def __init__(self):
        # initialize class variables
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
        self.jointPub = 0
        self.stiffness = False  

        pass


    def key_cb(self,data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    def joints_cb(self,data):
        #rospy.loginfo("joint states "+str(data.name)+str(data.position))
        # store current joint information in class variables
        self.joint_names = data.name 
        self.joint_angles = data.position
        self.joint_velocities = data.velocity

        pass

    def bumper_cb(self,data):
        rospy.loginfo("bumper: "+str(data.bumper)+" state: "+str(data.state))
        if data.bumper == 0:
            self.stiffness = True
        elif data.bumper == 1:
            self.stiffness = False

    def touch_cb(self,data):
        # self.set_stiffness(False)
        # rospy.sleep(3.0)
        print("activated")
        rospy.loginfo("touch button: "+str(data.button)+" state: "+str(data.state))
        if data.button == 1 and data.state == 1: #TB1
            self.set_stiffness(True)
            print("TB1 Head activated")
            self.set_joint_angles(2.0, "RShoulderPitch")
            self.set_joint_angles(0.8, "RElbowYaw")
            #rospy.sleep(3.0)
            #self.set_stiffness(False)

        elif data.button == 2 and data.state == 1:
            self.set_stiffness(True)
            print("TB2 head activated")
            #self.set_joint_angles(2.0, "RShoulderPitch")
            self.set_joint_angles(0.8, "LElbowYaw")
            self.set_joint_angles(0.2, "LElbowYaw")
            rospy.sleep(1.0)
            #self.set_joint_angles(0.5, "RShoulderPitch")
            self.set_joint_angles(0.8, "LElbowYaw")
            rospy.sleep(1.0)
            self.set_joint_angles(0.2, "LElbowYaw")
            rospy.sleep(1.0)
            self.set_joint_angles(0.8, "LElbowYaw")           
            rospy.sleep(3.0)
            


        elif data.button == 3 and data.state == 1: #TB3
            self.set_stiffness(True)
            print("TB3 activated")
            self.set_joint_angles(2.0, "RShoulderPitch")
            #self.set_joint_angles(0.8, "LShoulderPitch")
            #self.set_joint_angles(0.8, "LElbowYaw")
            self.set_joint_angles(0.8, "RElbowYaw")
            self.set_joint_angles(0.8, "LElbowYaw")
            
            #self.set_joint_angles(0.2, "RElbowYaw")
            rospy.sleep(1.0)
            #self.set_joint_angles(0.5, "RShoulderPitch")
            self.set_joint_angles(0.2, "LElbowYaw")
            self.set_joint_angles(0.2, "RElbowYaw")
            #self.set_joint_angles(0.9, "RElbowYaw")
            rospy.sleep(1.0)
            self.set_joint_angles(0.8, "RElbowYaw")
            self.set_joint_angles(0.8, "LElbowYaw")
            #self.set_joint_angles(0.2, "RElbowYaw")
            rospy.sleep(1.0)
            self.set_joint_angles(0.2, "LElbowYaw")
            self.set_joint_angles(0.2, "RElbowYaw")
            #self.set_joint_angles(0.9, "RElbowYaw")
            rospy.sleep(1.0)
            self.set_joint_angles(0.8, "RElbowYaw")
            self.set_joint_angles(0.8, "LElbowYaw")
            #self.set_joint_angles(0.2, "RElbowYaw")
            rospy.sleep(3.0)
            #self.set_stiffness(False)



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
            NBD.blob_detection(src) #[hkh]
            # cv2.imshow("Keypoints", cv_image) #Keypoints are already implemented above

            cv2.waitKey(3)

        except CvBridgeError as e:
            rospy.logerr(e)
        
       

    # sets the stiffness for all joints. can be refined to only toggle single joints, set values between [0,1] etc
    def set_stiffness(self,value):
        if value == True:
            service_name = '/body_stiffness/enable'
        elif value == False:
            service_name = '/body_stiffness/disable'
        try:
            stiffness_service = rospy.ServiceProxy(service_name,Empty)
            stiffness_service()
        except rospy.ServiceException, e:
            rospy.logerr(e)

    def set_joint_angles(self, head_angle, topic):

        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(topic) # each joint has a specific name, look into the joint_state topic or google  # When I
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.1 # keep this low if you can
        print(str(joint_angles_to_set))
        self.jointPub.publish(joint_angles_to_set)
        


    def central_execute(self):
        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name
        self.set_stiffness(True)
        # create several topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw", Image, self.image_cb)
        self.jointPub = rospy.Publisher("joint_angles", JointAnglesWithSpeed, queue_size=10)


        # test sequence to demonstrate setting joint angles
        # self.set_stiffness(True) # don't let the robot stay enabled for too long, the motors will overheat!! (don't go for lunch or something)
        
        
        #rospy.sleep(1.0)
        #self.set_joint_angles(0.5)
        #rospy.sleep(3.0)
        #self.set_joint_angles(0.0)
        # rospy.sleep(3.0)
        #self.set_stiffness(False) # always check that your robot is in a stable position before disabling the stiffness!!

        rate = rospy.Rate(10) # sets the sleep time to 10ms

        # while not rospy.is_shutdown():
        #     self.set_stiffness(self.stiffness)
        #     rate.sleep()

        # rospy.spin() just blocks the code from exiting, if you need to do any periodic tasks use the above loop
        # each Subscriber is handled in its own thread
        rospy.spin()

if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()

#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2

class tutorial3:

    def __init__(self):
        self.blobX = 0
        self.blobY = 0
        self.shoulderRoll = 0
        self.shoulderPitch = 0
        #self.motionProxy = self.getProxy("ALMotion")

        # For setting the stiffnes of single joints
        self.stiffnesPub = 0
        

    # For reading in touch on TB1
    # also collect training data here(?)
    def touch_cb(self,data):
        rospy.loginfo("touch button: "+str(data.button)+" state: "+str(data.state))
        if data.button == 1 and data.state == 1:
            # save those values to a file
            self.shoulderRoll
            self.shoulderPitch
            self.blobX
            self.blobY


    # for setting initial stiffness values
    def set_stiffness(self,value):
        if value == True:
            service_name = '/body_stiffness/enable'
        elif value == False:
            service_name = '/body_stiffness/disable'
        try:
            stiffness_service = rospy.ServiceProxy(service_name,Empty)
            stiffness_service()
        except rospy.ServiceException as e:
            rospy.logerr(e)

    def joints_cb(self,data):
        for jointNames in data:
            if jointNames.name == "RShoulderPitch":
                self.shoulderPitch = jointNames.position
                #rospy.loginfo("save")
            if jointNames.name == "RShoulderRoll":
                self.shoulderRoll = jointNames.position
            


    def tutorial3_execute(self):
        rospy.init_node('tutorial3_node',anonymous=True) #initilizes node, sets name

        # create several topic subscribers
        rospy.Subscriber("joint_states",JointAnglesWithSpeed,self.joints_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)

        self.stiffnesPub = rospy.Publisher("joint_stiffness", JointState, queue_size=10)


        # Set the initial stiffnesses
        stiffnessState = JointState()

        stiffnessState.msg = "HeadYaw"    #TODO: Head topic 
        stiffnessState.effort = 0.9
        self.stiffnesPub.publish(stiffnessState)

        stiffnessState.msg = "HeadPitch"    
        stiffnessState.effort = 0.9
        self.stiffnesPub.publish(stiffnessState)

        stiffnessState.msg = "RShoulderPitch"    
        stiffnessState.effort = 0
        self.stiffnesPub.publish(stiffnessState)

        stiffnessState.msg = "RShoulderRoll"    
        stiffnessState.effort = 0
        self.stiffnesPub.publish(stiffnessState)

        rospy.spin()
    




if __name__=='__main__':
    # instantiate class and start loop function
    node_instance = tutorial3()
    node_instance.tutorial3_execute()


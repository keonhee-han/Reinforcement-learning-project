#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
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
            br = CvBridge()
 
            # Output debugging information to the terminal
            rospy.loginfo("receiving video frame")
            
            # Convert ROS Image message to OpenCV image
            current_frame = br.imgmsg_to_cv2(data)
            current_frame = cv2.cvtColor(current_frame,cv2.COLOR_BGR2HSV)

            params = cv2.SimpleBlobDetector_Params()

            # Change thresholds
            params.minThreshold = 1
            params.maxThreshold = 255


            # Filter by Area.
            params.filterByArea = True
            params.minArea = 30

            # Filter by Circularity
            params.filterByCircularity = True
            params.minCircularity = 0.1

            params.filterByColor = True
            params.blobColor = 255

            
            lower_green = np.array([160,100,20])
            upper_green = np.array([179,255,255])
            # Threshold the HSV image to get only blue colors
            mask = cv2.inRange(current_frame, lower_green, upper_green)
            
            erode_kernel = np.ones((3,3),np.uint8)
            eroded_img = cv2.erode(mask,erode_kernel,iterations = 1)
        
            # dilate
            dilate_kernel = np.ones((10,10),np.uint8)
            dilate_img = cv2.dilate(eroded_img,dilate_kernel,iterations = 1)
            detector = cv2.SimpleBlobDetector_create(params)
            # Detect blobs.
            
            # Create a detector with the parameters
            # OLD: detector = cv2.SimpleBlobDetector(params)
            detector = cv2.SimpleBlobDetector_create(params)


            # Detect blobs.
            #keypoints = detector.detect(image_hsv)
            keypoints = detector.detect(dilate_img)
            
            if(len(keypoints) >= 0):
                max = 0
                xCoord = 0
                yCoord = 0
                maxObject = None
                for blob in keypoints:
                    if(blob.size>max):
                        max = blob.size
                        xCoord = blob.pt[0]
                        yCoord = blob.pt[1]
                        maxObject = blob

            # Round the coordinates to get pixel coordinates:
            xPixel = round(xCoord)
            yPixel = round(yCoord)

            # For better readability, round size to 3 decimal indices
            blobSize = round(max, 3)

            rospy.loginfo("Biggest blob: x Coord: " + str(xPixel) + " y Coord: " + str(yPixel) + " Size: " + str(blobSize))

            #im_with_keypoints = cv2.drawKeypoints(current_frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.circle(frame,(int(kp_max.pt[0]),int(kp_max.pt[1])),int(kp_max.size),(0,255,0),2)

            cv2.imshow("Keypoints", im_with_keypoints)
            
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
        except rospy.ServiceException as e:
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
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10)


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

#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import csv

class tutorial3:

    def __init__(self):
        self.blobX = 0
        self.blobY = 0
        self.blobSize = 0
        self.shoulderRoll = 0
        self.shoulderPitch = 0
        # For setting the stiffnes of single joints
        self.stiffnesPub = 0

    def image_cb(self,data):
        bridge_instance = CvBridge()
        try:
            br = CvBridge()
 
            # Output debugging information to the terminal
            #rospy.loginfo("receiving video frame")
            
            # Convert ROS Image message to OpenCV image
            current_frame_update = br.imgmsg_to_cv2(data)
            current_frame = cv2.bilateralFilter(current_frame_update,15,75,75)
            current_frame = cv2.cvtColor(current_frame,cv2.COLOR_BGR2HSV)

            params = cv2.SimpleBlobDetector_Params()

            # Change thresholds
            params.minThreshold = 1
            params.maxThreshold = 255


            # Filter by Area.
            params.filterByArea = True
            params.minArea = 15

            # Filter by Circularity
            params.filterByCircularity = False
            params.minCircularity = 0.1

            params.filterByColor = False
            params.blobColor = 255

            
            lower_red = np.array([160,100,20])
            upper_red = np.array([179,255,255])
            # Threshold the HSV image to get only blue colors
            mask2 = cv2.inRange(current_frame, lower_red, upper_red)
            
            lower_red = np.array([0,100,20])
            upper_red = np.array([10,255,255])
            # Threshold the HSV image to get only blue colors
            mask1 = cv2.inRange(current_frame, lower_red, upper_red)

            mask = mask1 +mask2
            dilade_kernel = np.ones((10,10),np.uint8)
            dilated_img = cv2.dilate(mask,dilade_kernel,iterations = 1)
        
            # dilate
            erosion_kernel = np.ones((10,10),np.uint8)
            closing_img = cv2.erode(dilated_img,erosion_kernel,iterations = 1)
            detector = cv2.SimpleBlobDetector_create(params)
            # Detect blobs.
            
            # Create a detector with the parameters
            # OLD: detector = cv2.SimpleBlobDetector(params)
            detector = cv2.SimpleBlobDetector_create(params)


            # Detect blobs.
            #keypoints = detector.detect(image_hsv)
            keypoints = detector.detect(closing_img)
            max = 0
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

            
            self.blobX = xPixel
            self.blobY = yPixel
            self.blobSize = max

            # For better readability, round size to 3 decimal indices
            blobSize = round(max, 3)

            #rospy.loginfo("Biggest blob: x Coord: " + str(xPixel) + " y Coord: " + str(yPixel) + " Size: " + str(blobSize))

            im_with_keypoints = cv2.drawKeypoints(current_frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            #cv2.circle(frame,(int(kp_max.pt[0]),int(kp_max.pt[1])),int(kp_max.size),(0,255,0),2)

            cv2.imshow("Keypoints", im_with_keypoints)
            
            cv2.waitKey(3)
           
        except CvBridgeError as e:
            rospy.logerr(e)
        

    # For reading in touch on TB1
    # also collect training data here(?)
    def touch_cb(self,data):
        #rospy.loginfo("touch button: "+str(data.button)+" state: "+str(data.state))
        if data.button == 1 and data.state == 1:
            # save those values to a file
            # Check if blob has a size greater 0:
            # Save current variable value to not have it change again due to callback
            saveX = self.blobX
            saveY = self.blobY
            savePitch = self.shoulderPitch
            saveRoll = self.shoulderRoll
            size = self.blobSize
            if size > 0:
                rospy.loginfo("----- wrote to file -------")
                # open the file in the write mode
                f = open("/home/bio/Desktop/BIHR_Nao2022/src/tutorial_3/scripts/samples.csv", "a")
                writer = csv.writer(f)
                row = [saveX, saveY, savePitch, saveRoll]
                rospy.loginfo(row)
                writer.writerow(row)
                f.flush()
                f.close()



    def joints_cb(self,data):
        #rospy.loginfo("entering joint states")
        for index, jointNames in enumerate(data.name):
            if jointNames == "RShoulderPitch":
                self.shoulderPitch = data.position[index]
                #rospy.loginfo(data.position[index])
                continue
                #rospy.loginfo("save")
            elif jointNames == "RShoulderRoll":
                self.shoulderRoll = data.position[index]
            


    def tutorial3_execute(self):
        rospy.init_node('tutorial3_node',anonymous=True) #initilizes node, sets name

        # create several topic subscribers
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)

        self.stiffnesPub = rospy.Publisher("joint_stiffness", JointState, queue_size=10)

        # Set the initial stiffnesses
        stiffnessState = JointState()

        stiffnessState.name = "HeadYaw"
        stiffnessState.effort = 0.9
        self.stiffnesPub.publish(stiffnessState)

        stiffnessState.name = "HeadPitch"    
        stiffnessState.effort = 0.9
        self.stiffnesPub.publish(stiffnessState)
       
        stiffnessState.name = "RShoulderPitch"  
        stiffnessState.effort = 0
        self.stiffnesPub.publish(stiffnessState)

        stiffnessState.name = "RShoulderRoll"
        stiffnessState.effort = 0
        self.stiffnesPub.publish(stiffnessState)

        rospy.spin()
    




if __name__=='__main__':
    # instantiate class and start loop function
    node_instance = tutorial3()
    node_instance.tutorial3_execute()


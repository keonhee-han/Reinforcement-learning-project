#!/usr/bin/env python
from __future__ import print_function
import cv2
import numpy as np
import time
import urllib
import Algorithms as AG
import matplotlib.pyplot as plt

import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

MV = ''
fn = ''
fn_flag = 0
ISM = 0
H = 0
url = ""


# def blob_detection(self, src, *arg):
        # h0 = cv2.getTrackbarPos('h min', 'control')
        # h1 = cv2.getTrackbarPos('h max', 'control')
        # s0 = cv2.getTrackbarPos('s min', 'control')
        # s1 = cv2.getTrackbarPos('s max', 'control')
        # v0 = cv2.getTrackbarPos('v min', 'control')
        # v1 = cv2.getTrackbarPos('v max', 'control')

def blob_detection(src):
        IHeight, IWidth, _ = src.shape  # assigning image info
        # src = br.imgmsg_to_cv2(data)
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        print("---------------------------------\n image shape:", src.shape)
        lower = np.array((0, 20, 221))
        upper = np.array((40, 255, 225))
        mask = cv2.inRange(hsv, lower, upper)
        '''Gaussian blur'''
        kernel = np.ones((7, 7), np.uint8)  # np.uint8 = Byte (-128 to 127) : black white color range
        erode = cv2.erode(mask, kernel, iterations=1)  # morphological transformation with mask
        dilate = cv2.dilate(erode, kernel,
                            iterations=2)  # since the white noise removed, the size is small but dialatation is needed
        filtered_EDG = cv2.GaussianBlur(dilate, (5, 5), 2)
        mask_final = dilate

        # Apply mask to original image, show results
        # img_result = cv2.bitwise_and(src, src, mask=mask)
        img_result = cv2.bitwise_and(src, src, mask=mask_final)
        cv2.imshow('mask', mask_final)
        cv2.imshow('image seen through mask', img_result)

        # Parameter definition for SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 1000
        params.maxArea = 200000
        params.filterByInertia = True
        params.minInertiaRatio = 0.0
        params.maxInertiaRatio = 0.8

        # Applying the params
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(~mask_final)

        # draw keypoints
        im_with_keypoints = cv2.drawKeypoints(~mask_final, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Keypoints", im_with_keypoints)


        '''Contours'''
        ret, contours, hierarchy = cv2.findContours(filtered_EDG, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Listing Hierarchy structures of contours and generating coordinates for recognized shapes
        total_contours = len(contours)  # Total amount of recognized weeds

        # Fidining the Max contour
        maxContour = 0
        maxContourData = None
        for contour in contours:
            contourSize = cv2.contourArea(contour)
            if contourSize > maxContour:
                maxContour = contourSize
                maxContourData = contour
        
        if maxContourData == None:
            rospy.loginfo("maxContourData set to None!")
            return

        ## Draw
        cv2.drawContours(src, maxContourData, -1, (0, 255, 0), 2, lineType=cv2.LINE_4)
        cv2.imshow('image with countours', src)

        #https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
        # Calculate image moments of the detected contour = (Center of blobs)
        M = cv2.moments(maxContourData)

        try:
            # Draw a circle based centered at centroid coordinates
            xPixel = int(M['m10'] / M['m00'])
            yPixel = int(M['m01'] / M['m00'])
            cv2.circle(src, ( xPixel, yPixel ), 5, (0, 0, 0), -1)
            rospy.loginfo("Biggest blob: x Coord: " + str(xPixel) + " y Coord: " + str(yPixel) + " Size: " + str(blobSize))

            # Show image:
            cv2.imshow("outline contour & centroid", src)

        except ZeroDivisionError:
            pass

# def receive_message():
#     cv2.namedWindow('Enable_fun', 0)
#     #Enable executing real time weed's center detection with Android camera
#     cv2.createTrackbar('E_WD4D', 'Enable_fun', 0, 1, callback)  # Enable Weed Detection for Driving
#
#     cv2.namedWindow('control', 0)
#     cv2.createTrackbar('Resize', 'control', 3, 10, callback)
#     cv2.createTrackbar('h min', 'control', 20, 179, callback)
#     cv2.createTrackbar('h max', 'control', 40, 179, callback)
#     cv2.createTrackbar('s min', 'control', 20, 255, callback)
#     cv2.createTrackbar('s max', 'control', 255, 255, callback)
#     cv2.createTrackbar('v min', 'control', 221, 255, callback)
#     cv2.createTrackbar('v max', 'control', 255, 255, callback)
#
#     # Tells rospy the name of the node.
#     # Anonymous = True makes sure the node has a unique name. Random
#     # numbers are added to the end of the name.
#     rospy.init_node('webcam_sub_py', anonymous=True)
#
#     # Node is subscribing to the video_frames topic
#     rospy.Subscriber('video_frames', Image, callback)
#
#     # spin() simply keeps python from exiting until this node is stopped
#     rospy.spin()
#     # while 1:
#     #     if cv2.waitKey(1) & 0xFF == ord('q'):
#     #         cv2.destroyAllWindows()
#     #         break;
#
#
# if __name__ == '__main__':
#     receive_message()
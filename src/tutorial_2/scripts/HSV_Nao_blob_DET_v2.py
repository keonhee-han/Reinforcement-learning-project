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


def callback(self, src, *arg):
    global MV
    global fn
    global fn_flag
    global ISM
    global H
    global url

    try:
        br = CvBridge()

        # Output debugging information to the terminal
        rospy.loginfo("receiving video frame")

        if (fn_flag == 1):
            src = cv2.imread(fn)

        elif (fn_flag == 0):
            fn = input(
                "Input image file including extension to test the program : \n eg) img/D7.jpg \n Press 'Q' to exit the current program. \n")
            fn_flag = 1
            src = cv2.imread(fn)

        start = time.time()  # start Image process time counting..

        E_WD4D = cv2.getTrackbarPos('E_WD4D', 'Enable_fun')
        # Resize = cv2.getTrackbarPos('Resize', 'control')
        h0 = cv2.getTrackbarPos('h min', 'control')
        h1 = cv2.getTrackbarPos('h max', 'control')
        s0 = cv2.getTrackbarPos('s min', 'control')
        s1 = cv2.getTrackbarPos('s max', 'control')
        v0 = cv2.getTrackbarPos('v min', 'control')
        v1 = cv2.getTrackbarPos('v max', 'control')
        CAMin = cv2.getTrackbarPos('Con_Area min', 'control')  # Counter area from min to max
        CAMax = cv2.getTrackbarPos('Con_Area max', 'control')
        DET_offset = cv2.getTrackbarPos('DET_offset', 'control')

        # src = cv2.resize(src, None, fx=(Resize / 10), fy=(Resize / 10), interpolation=cv2.INTER_CUBIC)  # resizing image

        IHeight, IWidth, _ = src.shape  # assigning image info

        hsv = br.imgmsg_to_cv2(data) 

        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

        print("---------------------------------")
        print("resized image shape:", src.shape)

        lower = np.array((h0, s0, v0))
        upper = np.array((h1, s1, v1))
        mask = cv2.inRange(hsv, lower, upper)

        filtered_E = AG.filter_E(mask)
        filtered_ED = AG.filter_ED(mask)
        filtered_EDG = AG.filter_EDG(mask)

        img_result = cv2.bitwise_and(src, src, mask=mask)

        '''Contours'''
        # ret, contours, hierarchy = cv2.findContours(filtered_EDG, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(filtered_EDG, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Listing Hierarchy structures of contours and generating coordinates for recognized shapes
        total_contours = len(contours)  # Total amount of recognized weeds
        result = src.copy()

        total_area = 0
        contours_error = []
        contours_ok = []
        W_area = []
        W_area_argmax = 0
        W_Compare_Area = []
        WeedNumber = 0
        MV = ""
        DET = 0
        font = cv2.FONT_HERSHEY_SIMPLEX  # font assignment

        for cnt in contours:
            WeedNumber += 1
            area = cv2.contourArea(cnt)
            total_area += area
            if (area > CAMin and area < CAMax):
                W_Compare_Area.append(area)
                print("Weed #" + str(WeedNumber) + " area : " + str(area))
                W_area.append(area)
                contours_ok.append(cnt)
            else:
                print("ERROR AREA: ", area)
                contours_error.append(cnt)

        if len(W_area) > 0:
            W_area_argmax = np.argmax(W_area) + 1
            print("Weed_area_list : ", W_area)
            print("Weed_argmax : Weed #", W_area_argmax)

        if len(contours) > 0:
            media = total_area / len(contours)
            print("\tAREA MEDIA : %.2f" % media)  # average area of calculated weeds, including error area

        DET_L = -50 - DET_offset
        DET_R = 50 + DET_offset

        '''Draw contours'''
        WeedNumber = 0  # Reassignment for counting weed numbers again.
        for crc in contours_ok:
            WeedNumber += 1
            print("Weed #%d" % WeedNumber)
            # print("area : ", contours_ok[WeedNumber])
            (x, y), radius = cv2.minEnclosingCircle(crc)
            center = (int(x), int(y))
            DET = int(x) - IWidth // 2

            radius = int(radius)
            # Dis = AG.calculateDistance(IWidth//2, IHeight//2, int(x),int(y)) #dis from center of cam
            Dis = AG.calculateDistance(IWidth // 2, IHeight // 2, int(x), int(y))  # dis from (0,0) cam

            '''Enable Weed Center for Driving'''
            if (E_WD4D == 1):
                cv2.circle(result, center, radius, (0, 250, 0), 2)  # Weed Center's point drawn
                cv2.circle(result, center, 3, (250, 0, 250), 2)
                cv2.line(result, (IWidth // 2, IHeight // 2), center, (255, 0, 0), 2)
                cv2.putText(result, 'Weed#%d, Dis : %d' % (WeedNumber, Dis), (int(x), int(y)), font, 0.4, (255, 255, 255),
                            1, cv2.LINE_AA)

                print("Center coordinate : ", center)
                print("Distance from center : %d" % Dis)
                print("Weed x axis distance from screen : %d" % DET) 
                # Round the coordinates to get pixel coordinates:
                xPixel = x
                yPixel = y
                rospy.loginfo("Biggest blob: x Coord: " + str(xPixel) + " y Coord: " + str(yPixel) + " Size: " + str(blobSize))

        '''Process ending time'''
        stop = time.time()
        diff = stop - start  
        t = str("%.3f" % diff)
        # fps = str(int(1//diff))
        # text = "t["+t+"] fps:["+fps+"] AREAS:["+str(len(contours_ok))+"]" + "Movement:["+ MV +"]"

        text = "t[" + t + "] AREAS:[" + str(len(contours_ok)) + "]"
        # Total amount of work time for image process
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, text, (15, 30), font, .5, (255, 255, 255), 2)
        # font assignment for result

        # show
        print("\tTOTAL: ", total_contours)
        print("\tOK: ", len(contours_ok))
        print("\tERRORS: ", len(contours_error))

        cv2.imshow('original', src)
        if (E_WD4D == 1):
            cv2.imshow('mask', mask)
            cv2.imshow('filter_Erosion: ', filtered_E)
            cv2.imshow('filter_Erosion + Dilation: ', filtered_ED)
            cv2.imshow('filter_Erosion + Dilation + Gaussian_blur: ', filtered_EDG)
            cv2.imshow('result', result)
            cv2.imshow('Bitwise_and', img_result)

        cv2.waitKey(3)


    except CvBridgeError as e:
        rospy.logerr(e)

def receive_message():
    cv2.namedWindow('Enable_fun', 0)
    #Enable executing real time weed's center detection with Android camera
    cv2.createTrackbar('E_WD4D', 'Enable_fun', 0, 1, callback)  # Enable Weed Detection for Driving

    cv2.namedWindow('control', 0)
    cv2.createTrackbar('Resize', 'control', 3, 10, callback)
    cv2.createTrackbar('h min', 'control', 20, 179, callback)
    cv2.createTrackbar('h max', 'control', 40, 179, callback)
    cv2.createTrackbar('s min', 'control', 20, 255, callback)
    cv2.createTrackbar('s max', 'control', 255, 255, callback)
    cv2.createTrackbar('v min', 'control', 221, 255, callback)
    cv2.createTrackbar('v max', 'control', 255, 255, callback)
    cv2.createTrackbar('Con_Area min', 'control', 0, 52500, callback)
    cv2.createTrackbar('Con_Area max', 'control', 52500, 52500, callback)
    cv2.createTrackbar('DET_offset', 'control', 0, 200, callback)

    # Tells rospy the name of the node.
    # Anonymous = True makes sure the node has a unique name. Random
    # numbers are added to the end of the name.
    rospy.init_node('webcam_sub_py', anonymous=True)

    # Node is subscribing to the video_frames topic
    rospy.Subscriber('video_frames', Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    # while 1:
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         cv2.destroyAllWindows()
    #         break;


if __name__ == '__main__':
    receive_message()
#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class CMAC():
    def __init__(self, input_size = 10, RF_size = 5):
        # initialize class variables
        # Layer1: quantize the inputs
        self.y1_ = np.array([input_size]) # inputs size for 1st array
        self.y2_ = np.array([input_size]) # inputs size for 2nd array
        self.z1_ = (0,0) # feature detecting neurons for each input at y1_
        self.z2_ = (0,0) # feature detecting neurons for each input at y2_
        self.n_a = RF_size    # size of receptive field

        # Layer2: AND logic
        self.a_ = np.zeros([2, z1_[1], z2_[1]]) # association neurons: AND logic from both input neurons (z_1i, z_2j)
        self.n_v =  # number of associative neurons

        # Layer3: weighted sum of output neurons
        self.w_ = np.zeros([2, z1_[1], z2_[1]]) # creating weights matrix to be updated (2xjxk)
        self.nx_ = (0,0)    # 2D pixel indices as input, size = 2
        self.ny_ = (0,0)    # 2 joints as output
        self.x_ = None

    def output_calc_x(self, w_):
        self.x_ = np.sum(w_* self.a_, axis=1).sum(axis=1)   # Summing all elements along output axis -> (2x1) as output dim.

if __name__=='__main__':
    # instantiate class and start loop function
    CMAC_instance = CMAC(10, 5)
    CMAC_instance.central_execute()
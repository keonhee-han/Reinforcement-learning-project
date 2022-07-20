#!/usr/bin/env python
import random
import rospy
from std_msgs.msg import String, Header
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed, Bumper, HeadTouch
from naoqi import ALProxy
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import csv
import random
from naoqi import ALProxy
import sys
import RL_DT
from sklearn import tree
import csv
import copy
import cv2.aruco as aruco
import argparse


class tutorial5_soccer:
    def __init__(self, init_joint=0, init_goal_keeper=0, gamma=0.8, MAXSTEPS=100, goal_keeper_num = 3):
        self.blobX = 0
        self.blobY = 0
        self.blobSize = 0
        self.shoulderRoll = 0
        self.shoulderPitch = 0
        # For setting the stiffnes of single joints
        self.jointPub = 0
        self.kick_reward = 0

        self.init_state = init_joint
        self.init_goal_keeper = init_goal_keeper

        self.state = 0 # for RL-DT to read
        self.state_prime = 0 # for RL-DT to read
        self.action = 0 # for RL-DT to read
        self.instant_reward = 0
        self.cumulative_reward = []

        # for RL-DT
        self.A = [0, 1, 2]  # 'Left': 0, 'Right': 1, 'Kick': 2
        self.sM = []  # set of all state
        self.visit = np.zeros((goal_keeper_num,10, 3))  # counting the amount of visited state
        self.Q = np.zeros((goal_keeper_num,10, 3))  # q table
        self.Rm = np.zeros((goal_keeper_num, 10, 3))  # reward matrix
        self.Ch = False
        self.exp = False
        self.gamma = gamma
        self.maxstep = MAXSTEPS

        self.possible_state = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.X_train = []
        self.y_train = []
        self.rewardTree = tree.DecisionTreeClassifier()
        self.max_marker_distance = 226
        self.marker_id_left_post = 1
        self.marker_id_right_post = 2
        self.marker_id_goalkeeper = 77
        self.x_left_post = 0
        self.x_right_post = 0
        self.x_goal_keeper = 0
        self.goalkeeper_state = 0
        self.ARUCO_DICT = {
            "DICT_4X4_50": aruco.DICT_4X4_50,
            "DICT_4X4_100": aruco.DICT_4X4_100,
            "DICT_4X4_250": aruco.DICT_4X4_250,
            "DICT_4X4_1000": aruco.DICT_4X4_1000,
            "DICT_5X5_50": aruco.DICT_5X5_50,
            "DICT_5X5_100": aruco.DICT_5X5_100,
            "DICT_5X5_250": aruco.DICT_5X5_250,
            "DICT_5X5_1000": aruco.DICT_5X5_1000,
            "DICT_6X6_50": aruco.DICT_6X6_50,
            "DICT_6X6_100": aruco.DICT_6X6_100,
            "DICT_6X6_250": aruco.DICT_6X6_250,
            "DICT_6X6_1000": aruco.DICT_6X6_1000,
            "DICT_7X7_50": aruco.DICT_7X7_50,
            "DICT_7X7_100": aruco.DICT_7X7_100,
            "DICT_7X7_250": aruco.DICT_7X7_250,
            "DICT_7X7_1000": aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL
            #"DICT_APRILTAG_16h5": aruco.DICT_APRILTAG_16h5,
            #"DICT_APRILTAG_25h9": aruco.DICT_APRILTAG_25h9,
            #"DICT_APRILTAG_36h10": aruco.DICT_APRILTAG_36h10,
            #"DICT_APRILTAG_36h11": aruco.DICT_APRILTAG_36h11
            }

    # Callback function for reading in the joint values
    def joints_cb(self, data):
        # rospy.loginfo("joint states "+str(data.name)+str(data.position))
        # store current joint information in class variables
        self.joint_names = data.name  # LHipRoll for move in or move out
        self.joint_angles = data.position
        self.joint_velocities = data.velocity

        pass

    # Read in the goal position!
    # TODO: Aruco marker detection
    def image_cb(self,data):
        bridge_instance = CvBridge()
        try:
            br = CvBridge()
            # Output debugging information to the terminal
            #rospy.loginfo("receiving video frame")
            # Convert ROS Image message to OpenCV image
            current_frame = br.imgmsg_to_cv2(data, "bgr8")
            args = self.args()
            #image = cv2.imread(args['image'])
            arucoDict = cv2.aruco.Dictionary_get(self.ARUCO_DICT[args['type']])
            arucoParams = cv2.aruco.DetectorParameters_create()
            (corners, ids, rejected) = cv2.aruco.detectMarkers(current_frame, arucoDict, parameters=arucoParams)

            if len(corners) == 3:
                for (markerCorner, markerId) in zip(corners, ids):
                    marker_id = markerId[0]
                    corners_abcd = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners_abcd
                    cX = int((topLeft[0] + bottomRight[0]) // 2)
                    #print(cX)
                    #print(markerId[0])
                    if(marker_id == self.marker_id_left_post):
                        self.x_left_post = cX
                    elif(marker_id == self.marker_id_right_post):
                        self.x_right_post = cX
                    elif(marker_id == self.marker_id_goalkeeper):
                        self.x_goal_keeper = cX

                distance = abs(self.x_left_post - self.x_goal_keeper)
                #print("Distance: ", distance)
                self.max_marker_distance = abs(self.x_left_post - self.x_right_post)
                #print("Goal-width: "+ str(self.max_marker_distance))
                self.goalkeeper_state = int(distance/float(self.max_marker_distance) * 3)
                #print("Goal_keeper_state: " + str(self.goalkeeper_state))


            if len(corners) > 0:
                ids = ids.flatten()
                for (markerCorner, markerId) in zip(corners, ids):
                    corners_abcd = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners_abcd
                    topRightPoint = (int(topRight[0]), int(topRight[1]))
                    topLeftPoint = (int(topLeft[0]), int(topLeft[1]))
                    bottomRightPoint = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeftPoint = (int(bottomLeft[0]), int(bottomLeft[1]))
                    cv2.line(current_frame, topLeftPoint, topRightPoint, (0, 255, 0), 2)
                    cv2.line(current_frame, topRightPoint, bottomRightPoint, (0, 255, 0), 2)
                    cv2.line(current_frame, bottomRightPoint, bottomLeftPoint, (0, 255, 0), 2)
                    cv2.line(current_frame, bottomLeftPoint, topLeftPoint, (0, 255, 0), 2)
                    cX = int((topLeft[0] + bottomRight[0]) // 2)
                    cY = int((topLeft[1] + bottomRight[1]) // 2)
                    self.goalkeeper_x = cX
                    self.goalkeeper_y = cY
                    cv2.circle(current_frame, (cX, cY), 4, (255, 0, 0), -1)
                    cv2.putText(current_frame, str(
                        int(markerId)), (int(topLeft[0] - 10), int(topLeft[1] - 10)), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 0, 255))  # print(arucoDict)
                    cv2.imshow("[INFO] marker detected", current_frame)

        except CvBridgeError as e:
            rospy.logerr(e)

    def args(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to detect")
        ap.add_argument("-l", "--length", type=float, default="0.09", help="length of the marker in meters")
        arguments = vars(ap.parse_args())
        return arguments

    def Arcuo_marker(self):
        return 0

    def goal_keeper_state(self):
        return 0

    def set_joint_angles(self, head_angle, topic):
        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(
            topic)  # each joint has a specific name, look into the joint_state topic or google  # When I
        joint_angles_to_set.joint_angles.append(
            head_angle)  # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False  # if true you can increment positions
        joint_angles_to_set.speed = 0.03  # keep this low if you can
        # print(str(joint_angles_to_set))
        self.jointPub.publish(joint_angles_to_set)

    def set_joint_angles_fast(self, head_angle, topic):
        # fast motion for kick!! careful
        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(
            topic)  # each joint has a specific name, look into the joint_state topic or google  # When I
        joint_angles_to_set.joint_angles.append(
            head_angle)  # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False  # if true you can increment positions
        joint_angles_to_set.speed = 0.6  # keep this low if you can
        # print(str(joint_angles_to_set))
        self.jointPub.publish(joint_angles_to_set)

    def set_joint_angles_list(self, head_angle_list, topic_list):
        # set the init one stand mode, doing it by all list together
        if len(head_angle_list) == len(topic_list):
            for i in range(len(topic_list)):
                head_angle = head_angle_list[i]
                topic = topic_list[i]
                joint_angles_to_set = JointAnglesWithSpeed()
                joint_angles_to_set.joint_names.append(
                    topic)  # each joint has a specific name, look into the joint_state topic or google  # When I
                joint_angles_to_set.joint_angles.append(
                    head_angle)  # the joint values have to be in the same order as the names!!
                joint_angles_to_set.relative = False  # if true you can increment positions
                joint_angles_to_set.speed = 0.1  # keep this low if you can
                # print(str(joint_angles_to_set))
                self.jointPub.publish(joint_angles_to_set)
                rospy.sleep(0.05)

# rosservice call /body_stiffness/disable "{}"
# optimal Q_Talbe for the middle goalkeeper
#     [[41.67488634  41.67192726  53.33990907]
#      [41.67192726  41.67003344  53.3375418]
#     [41.67003344
#     32.33602676
#     53.33602676]
#     [41.6688214   24.8688214   31.33505712]
#     [32.33505712
#     18.89505712
#     23.8680457]
#     [24.8680457   18.89687149  17.89443656]
#     [18.89443656
#     24.86949719
#     17.89559776]
#     [18.89559776  32.33559776  23.8684782]
#     [24.8684782
#     41.6684782
#     31.33478256]
#     [32.33478256  41.6684782   53.33478256]]

    def make_action(self, action):
        if action == 0:
            self.move_in()
            rospy.sleep(0.1)
        elif action == 1:
            self.move_out()
            rospy.sleep(0.1)
        else:
            self.kick()
            rospy.sleep(0.1)

    # Moves its left hip back and forward and then goes back into its initial position
    def kick(self):

        self.set_stiffness(True)
        # Move foot back
        self.set_joint_angles(0.48, "LHipPitch")
        rospy.sleep(1.0)
        # fast kick
        self.set_joint_angles_fast(-0.8, "LHipPitch")

        # Move the foot to original position
        rospy.sleep(2.0)
        # self.one_foot_stand()
        self.set_joint_angles(-0.3911280632019043, "LHipPitch")
        self.read_state_joint()

    def set_initial_stand(self):
        robotIP = '10.152.246.137'
        try:
            postureProxy = ALProxy('ALRobotPosture', robotIP, 9559)
        except Exception, e:
            print('could not create ALRobotPosture')
            print('Error was', e)
        postureProxy.goToPosture('Stand', 1.0)
        #print(postureProxy.getPostureFamily())

    def one_foot_stand(self):
        # it is the init state ready for kicking
        # careful !!!!!! very easy to fall
        print('one foot mode')
        self.set_stiffness(True)
        # using the rostopic echo to get the desired joint states, not perfect
        # rostopic echo /joint_states
        """
        position_ori = [0.004559993743896484, 0.5141273736953735, 1.8330880403518677, 0.19937801361083984, -1.9574260711669922,
            -1.5124820470809937, -0.8882279396057129, 0.32840001583099365, -0.13955211639404297, 0.31297802925109863,
            -0.3911280632019043, 1.4679961204528809, -0.8943638801574707, -0.12114405632019043, -0.13955211639404297,
            0.3697359561920166, 0.23772811889648438, -0.09232791513204575, 0.07980990409851074, -0.3282339572906494,
            1.676703929901123, -0.45717406272888184, 1.1964781284332275, 0.18872404098510742, 0.36965203285217285, 0.397599995136261]
        """

        # way1 the best position i find
        position = [0.004559993743896484, 0.5141273736953735, 1.8330880403518677, 0.19937801361083984,
                    -1.9574260711669922,
                    -1.5124820470809937, -0.8882279396057129, 0.32840001583099365, -0.13955211639404297, 0.48,
                    -0.3911280632019043, 1.2, -0.4, -0.12114405632019043, -0.13955211639404297,
                    0.3697359561920166, 0.23772811889648438, -0.09232791513204575, 0.07980990409851074,
                    -0.3282339572906494,
                    1.676703929901123, -0.8, 1.1964781284332275, 0.18872404098510742, 0.36965203285217285,
                    0.397599995136261]
        joints = ['HeadYaw', 'HeadPitch', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw',
                  'LElbowRoll', 'LWristYaw', 'LHand', 'LHipYawPitch', 'LHipRoll',
                  'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll', 'RHipYawPitch',
                  'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll',
                  'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand']
        '''
        # way 2
        position = [-0.015382051467895508, 0.5120565295219421, 1.8346221446990967, 0.1779019832611084, -1.937483787536621,
                    -1.5124820470809937, -1.0692400932312012, 0.32760000228881836, -0.11807608604431152, 0.31297802925109863,
                    -0.3911280632019043, 1.4664621353149414, -0.8943638801574707, -0.12114405632019043, -0.11807608604431152,
                    0.3697359561920166, 0.2530679702758789, -0.09232791513204575, -0.07665801048278809, -0.269942045211792,
                    1.6951122283935547, -0.4617760181427002, 1.1949440240859985, 0.2025299072265625, 0.3589141368865967, 0.39399999380111694]
        '''
        # backup init position
        # way3 [0.004559993743896484, 0.5141273736953735, 1.8330880403518677, 0.15335798263549805, -1.9129400253295898, -1.5032780170440674, -1.199629783630371, 0.32760000228881836, -0.22852396965026855, 0.4019498825073242, -0.3911280632019043, 1.4679961204528809, -0.8943638801574707, -0.12114405632019043, -0.22852396965026855, 0.3697359561920166, 0.23772811889648438, -0.09232791513204575, 0.08594608306884766, -0.3052239418029785, 1.6905097961425781, -0.44950389862060547, 1.1995460987091064, 0.1994619369506836, 0.3512439727783203, 0.3952000141143799]
        self.set_joint_angles_list(position, joints)

    def move_in(self):
        print('move in')
        self.instant_reward = -1
        if self.state != 0:
            self.state = self.state - 1
            print("dis state:", self.state)
            self.set_joint_LHipRoll(self.state)
        else:
            print("lower bound")

    def move_out(self):
        print('move out')
        self.instant_reward = -1
        if self.state != 9:
            self.state = self.state + 1
            print("dis state:", self.state)
            self.set_joint_LHipRoll(self.state)
        else:
            print("upper bound")

    def set_joint_LHipRoll(self, state):
        joint_angle = 0.48 + state * 0.03
        print('set_angle:', joint_angle)
        self.set_joint_angles(joint_angle, "LHipRoll")
        rospy.sleep(0.5)
        return self.read_state_joint()

    def read_state_joint(self):
        LHipRoll_angle = self.joint_angles[self.joint_names.index('LHipRoll')]
        state_joint = int((LHipRoll_angle - 0.48) / (0.75 - 0.48) * 10)
        print("LHipRoll:", LHipRoll_angle, "state_joint:", state_joint)
        return state_joint


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

    '''
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
    '''

    def touch_cb_reward(self, data):
        if data.button == 1 and data.state == 1:  # miss the goal
            print("miss")
            self.instant_reward = -2
            self.kick_reward = -2
            return self.instant_reward
        if data.button == 2 and data.state == 1:  # goal!!!
            print('goal')
            self.instant_reward = 20
            self.kick_reward = 20
            return self.instant_reward
        if data.button == 3 and data.state == 1:  # fall down
            print('fall down')
            self.instant_reward = -20
            self.kick_reward = -20
            return self.instant_reward
        else:
            #self.instant_reward = 0
            self.kick_reward = 0
            return False
        # how to build the waiting signal


    def touch_cb_test(self, data):
        # for test the movement
        if data.button == 1 and data.state == 1:  # TB1
            self.move_in()
        if data.button == 2 and data.state == 1:
            self.move_out()
        if data.button == 3 and data.state == 1:
            print("kick")
            self.kick()

    def tutorial5_soccer_execute_test_by_tactile(self):

        # cmac training here!!!
        rospy.init_node('tutorial5_soccer_node', anonymous=True)
        self.set_stiffness(True)
        self.jointPub = rospy.Publisher("joint_angles", JointAnglesWithSpeed, queue_size=10)

        # self.set_initial_stand()
        rospy.sleep(2.0)
        self.one_foot_stand()
        self.state = 0
        # rospy.Subscriber("joint_states",JointAnglesWithSpeed,self.joints_cb)
        rospy.Subscriber("tactile_touch", HeadTouch, self.touch_cb_test)
        rospy.Subscriber('joint_states', JointState, self.joints_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw", Image, self.image_cb)
        # start with setting the initial positions of head and right arm

        # rospy.Subscriber("/nao_robot/camera/top/camera/image_raw", Image, self.image_cb)

        rospy.spin()



    def State_Transition(self, state, action):
        shift = 0
        if action == 0:
            shift = -1
        elif action == 1:
            shift = 1
        next_state_joint = state[1] + shift # state[1] joint
        if next_state_joint < 0 or next_state_joint > 9:
            return [state[0], state[1]]
        return [state[0], next_state_joint]


    def get_predictions(self, s_m, a_m):
        r_pred = self.rewardTree.predict([[s_m[0],s_m[1], a_m]])
        print(r_pred)
        return r_pred[0]


    def add_experience_to_tree(self, s, action, r):
        self.X_train.append([s[0],s[1], action])
        self.y_train.append(r)
        self.rewardTree.fit(self.X_train, self.y_train)
        #print("fit!")
        return True


    def Update_Model(self,s,action,r, s_prime):
        # not completed
        # n = self.state_num
        self.Ch = self.add_experience_to_tree(s, action, r)
        # print("sM:",self.sM)
        for s_m in self.sM:
            for a_m in self.A:
                # print("pred:", s_m, a_m, self.get_predictions(s_m, a_m))
                # print("sm_am: ",s_m, a_m)
                self.Rm[s_m[0]][s_m[1]][a_m] = self.get_predictions(s_m, a_m)

        return self.Ch
                # self.Rm[s_m][a_m] = self.reward_true[s_m][a_m]


    def Check_Model(self):
        for r in np.nditer(self.Rm):
            if r > 0:
                return True
        return False



    def check_convergence(self, action_values_temp):
        for i in range(self.Q.shape[0]):
            for j in range(self.Q.shape[1]):
                for q in range(self.Q.shape[2]):
                    if (abs(self.Q[i][j][q] - action_values_temp[i][j][q]) > 0.01):
                        return False
        return True

    def Compute_Value(self, current_state, stepsize):
        # Value iteration
        # print("Compute_value")
        minivisits = np.min(self.visit[current_state[0]])
        #print("visit:", self.visit)
        converged = False
        while not converged:
            for step in range(stepsize):
                Q_temp = copy.deepcopy(self.Q)
                for s in self.sM:
                    for a in self.A:
                        if self.exp and self.visit[s[0]][s[1]][a] == minivisits:
                            # print("RMax")
                            self.Q[s[0]][s[1]][a] = 999
                        else:
                            # print("R")
                            self.Q[s[0]][s[1]][a] = self.Rm[s[0]][s[1]][a]
                            s_prime = self.State_Transition(s, a)
                            self.Q[s[0]][s[1]][a] += self.gamma*max(self.Q[s_prime[0]][s_prime[1]][:])
            # converged = self.check_convergence(Q_temp)
            converged = True

        return 0

    def q_max(self, state):
        # state[0] is goal keeper position. state[1] is joint angle
        Q = self.Q[state[0]][state[1]][:]
        print("Q: ",Q)
        max_q = Q[0]
        max_i = 0
        for i in range(len(Q)):
            if Q[i] > max_q:
                max_q = Q[i]
                max_i = i
        # print("max_action:", max_i)
        return max_i

    def test(self):
        rospy.init_node('tutorial5_soccer_node', anonymous=True)
        #rospy.Subscriber("tactile_touch", HeadTouch, self.touch_cb_reward)  # will give the data?
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw", Image, self.image_cb)

        rospy.spin()

    def tutorial5_soccer_joint_test(self):
        rospy.init_node('tutorial5_soccer_node', anonymous=True)
        self.set_stiffness(True)
        self.jointPub = rospy.Publisher("joint_angles", JointAnglesWithSpeed, queue_size=10)
        rospy.sleep(2.0)
        self.one_foot_stand()
        self.state = 0  # init state
        # rospy.Subscriber("joint_states",JointAnglesWithSpeed,self.joints_cb)
        rospy.Subscriber("tactile_touch", HeadTouch, self.touch_cb_test)
        rospy.Subscriber('joint_states', JointState, self.joints_cb)
        # start with setting the initial positions of head and right arm


        # rospy.Subscriber("/nao_robot/camera/top/camera/image_raw", Image, self.image_cb)

        rospy.spin()

    def tutorial5_soccer_train(self):
        rospy.init_node('tutorial5_soccer_node', anonymous=True)
        self.set_stiffness(True)
        self.jointPub = rospy.Publisher("joint_angles", JointAnglesWithSpeed, queue_size=10)
        rospy.sleep(2.0)
        self.one_foot_stand()
        # rospy.Subscriber("joint_states",JointAnglesWithSpeed,self.joints_cb)
        rospy.Subscriber("tactile_touch", HeadTouch, self.touch_cb_reward) # will give the data?
        rospy.Subscriber('joint_states', JointState, self.joints_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw", Image, self.image_cb)

        # self.state = 0  # init state
        s = [self.init_goal_keeper, self.init_state]
        temp = input("Please press any key to start learning")
        step = 0
        Goal_keeper = []
        while 0 not in Goal_keeper or 1 not in Goal_keeper or 2 not in Goal_keeper:
            print("Goalkeeper-state: "+str(self.goalkeeper_state))
            s[0] = self.goalkeeper_state
            Goal_keeper.append(s[0])
            self.sM.append(s)
            converged = False
            step = 0
            while not converged or np.min(self.visit[s[0]]) < 1:
            # while np.min(self.visit[s[0]]) < 2:
            # while step < self.maxstep:
                # print("visit_con:",self.visit[:])
                print("np_min:", np.min(self.visit[s[0]]))
                # break
            #while step<100:
                step = step + 1
                # print(converged)
                Q_temp = copy.deepcopy(self.Q)
                action = self.q_max(s)  # greedy action
                self.make_action(action)
                print("maxaction:", action)
                self.visit[s[0]][s[1]][action] += 1   # s[0] goal keeper, s[1] joint
                s_prime = self.State_Transition(s, action)
                print("s_prime:", s_prime)
                # r = rl_dt.reward_true[s][action]
                r = 0
                print("instant_kick:", self.kick_reward)
                if action == 0 or action == 1:
                    r = -1
                else:
                    print("wait for reward")
                    # wait reward signal after kick
                    r = input("reward:")  # hold on
                    # self.touch_cb_reward
                self.cumulative_reward.append(r)


                # return reward
                if s_prime not in self.sM:
                    self.sM.append(s_prime)

                self.Update_Model(s, action, r, s_prime) # update the reward tree, neglect transition tree
                # self.exp = self.Check_Model() # not use
                self.exp = True
                # print("exp:", self.exp)
                if np.min(self.visit) >= 1:
                    self.exp = False
                    # stop giving Rmax after every state is visited twice
                if self.Ch: # always true
                    # self.Compute_Value(300)
                    self.Compute_Value(s, 300)
                s = s_prime
                converged = self.check_convergence(Q_temp)
            #print(self.Q)
            #print(self.Rm)
            input("Please change the location of goal_keeper to continue learning:")

        rewards = self.cumulative_reward
        with open('cumulative_reward.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(self.cumulative_reward)
        input("Press any key for testing")

    def tutorial5_soccer_test(self):
        rospy.init_node('tutorial5_soccer_node', anonymous=True)
        # self.set_stiffness(True)
        self.jointPub = rospy.Publisher("joint_angles", JointAnglesWithSpeed, queue_size=10)
        rospy.sleep(2.0)
        # self.one_foot_stand()
        # rospy.Subscriber("joint_states",JointAnglesWithSpeed,self.joints_cb)
        rospy.Subscriber("tactile_touch", HeadTouch, self.touch_cb_reward)  # will give the data?
        rospy.Subscriber('joint_states', JointState, self.joints_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw", Image, self.image_cb)
        """
        self.Q = [[[  9.10098361,  12.62622951,  -9.89901639]
                  [  9.10098361,  17.03278689,  11.62622951]
                  [ 12.62622951,  22.54098361,  16.03278689]
                  [ 17.03278689,  29.42622951,  21.54098361]
                  [ 22.54098361,  22.54098361,  38.03278689]
                  [ 29.42622951,  17.03278689,  21.54098361]
                  [ 22.54098361,  12.62622951,  16.03278689]
                  [ 17.03278689,   9.10098361,  11.62622951]
                  [ 12.62622951,   6.28078689,   8.10098361]
                  [  9.10098361,   6.28078689, -12.71921311]]

                 [[  4.02462951,   6.28078689, -14.97537049]
                  [  4.02462951,   9.10098361,   5.28078689]
                  [  6.28078689,  12.62622951,   8.10098361]
                  [  9.10098361,  17.03278689,  11.62622951]
                  [ 12.62622951 , 22.54098361,  16.03278689]
                  [ 17.03278689,  29.42622951,  21.54098361]
                  [ 22.54098361,  22.54098361,  38.03278689]
                  [ 29.42622951,  17.03278689,  21.54098361]
                  [ 22.54098361,  12.62622951,  16.03278689]
                  [ 17.03278689,  12.62622951,  -6.37377049]]

                 [[  0.77576289,   2.21970361, -18.22423711]
                  [  0.77576289,   4.02462951,   1.21970361]
                  [  2.21970361,   6.28078689,   3.02462951]
                  [  4.02462951,   9.10098361,   5.28078689]
                  [  6.28078689,  12.62622951,   8.10098361]
                  [  9.10098361,  17.03278689,  11.62622951]
                  [ 12.62622951,  22.54098361,  16.03278689]
                  [ 17.03278689,  29.42622951,  21.54098361]
                  [ 22.54098361,  22.54098361,  38.03278689]
                  [ 29.42622951,  22.54098361,   3.54098361]]]
        """
        while True:
            input("Press any key to score a goal")
            goal_keeper = self.goalkeeper_state
            init_joint = np.random.choice(9)
            self.set_joint_LHipRoll(init_joint)

            # self.state = 0  # init state
            flag = True
            s = [goal_keeper, init_joint]
            while flag:
                action = self.q_max(s)
                self.make_action(action)
                if action == 2:
                    if input("Succeed or Fail?"):
                        flag = False
                s = self.State_Transition(s, action)

"""
Q table 
[[[ 24.86666667  32.33333333  23.86666667]
  [ 24.86666667  41.66666667  31.33333333]
  [ 32.33333333  41.66666667  53.33333333]
  [ 41.66666667  41.66666667  53.33333333]
  [ 41.66666667  41.66666667  53.33333333]
  [ 41.66666667  41.66666667  53.33333333]
  [ 41.66666667  41.66666667  53.33333333]
  [ 41.66666667  41.66666667  53.33333333]
  [ 41.66666667  41.66666667  53.33333333]
  [ 41.66666667  41.66666667  53.33333333]]

 [[ 41.66666667  32.33333333  53.33333333]
  [ 41.66666667  24.86666667  31.33333333]
  [ 32.33333333  18.89333333  23.86666667]
  [ 24.86666667  14.11466667  17.89333333]
  [ 18.89333333  18.89333333  13.11466667]
  [ 14.11466667  24.86666667  17.89333333]
  [ 18.89333333  32.33333333  23.86666667]
  [ 24.86666667  41.66666667  31.33333333]
  [ 32.33333333  41.66666667  53.33333333]
  [ 41.66666667  41.66666667  53.33333333]]

 [[ 41.66666667  41.66666667  53.33333333]
  [ 41.66666667  41.66666667  53.33333333]
  [ 41.66666667  41.66666667  53.33333333]
  [ 41.66666667  32.33333333  53.33333333]
  [ 41.66666667  24.86666667  31.33333333]
  [ 32.33333333  18.89333333  27.86666667]
  [ 24.86666667  14.11466667  17.89333333]
  [ 18.89333333  10.29173333  13.11466667]
  [ 14.11466667   7.23338667   9.29173333]
  [ 10.29173333   7.23338667   6.23338667]]]

learned reward
[[[ -1.  -1.  -2.]
  [ -1.  -1.  -2.]
  [ -1.  -1.  20.]
  [ -1.  -1.  20.]
  [ -1.  -1.  20.]
  [ -1.  -1.  20.]
  [ -1.  -1.  20.]
  [ -1.  -1.  20.]
  [ -1.  -1.  20.]
  [ -1.  -1.  20.]]

 [[ -1.  -1.  20.]
  [ -1.  -1.  -2.]
  [ -1.  -1.  -2.]
  [ -1.  -1.  -2.]
  [ -1.  -1.  -2.]
  [ -1.  -1.  -2.]
  [ -1.  -1.  -2.]
  [ -1.  -1.  -2.]
  [ -1.  -1.  20.]
  [ -1.  -1.  20.]]

 [[ -1.  -1.  20.]
  [ -1.  -1.  20.]
  [ -1.  -1.  20.]
  [ -1.  -1.  20.]
  [ -1.  -1.  -2.]
  [ -1.  -1.   2.]
  [ -1.  -1.  -2.]
  [ -1.  -1.  -2.]
  [ -1.  -1.  -2.]
  [ -1.  -1.  -2.]]]


('visit:', array([[[ 1.,  2.,  1.],
        [ 3.,  2.,  2.],
        [ 2.,  2.,  1.],
        [ 2.,  2.,  1.],
        [ 2.,  2.,  1.],
        [ 2.,  2.,  1.],
        [ 2.,  2.,  1.],
        [ 2.,  2.,  1.],
        [ 2.,  2.,  1.],
        [ 2.,  1.,  1.]],

       [[ 2.,  3.,  2.],
        [ 2.,  2.,  1.],
        [ 2.,  2.,  1.],
        [ 2.,  2.,  1.],
        [ 2.,  2.,  1.],
        [ 2.,  2.,  1.],
        [ 2.,  2.,  1.],
        [ 2.,  2.,  1.],
        [ 2.,  2.,  1.],
        [ 2.,  1.,  1.]],

       [[ 1.,  2.,  1.],
        [ 1.,  2.,  1.],
        [ 1.,  2.,  2.],
        [ 1.,  2.,  1.],
        [ 1.,  2.,  1.],
        [ 1.,  2.,  1.],
        [ 1.,  2.,  1.],
        [ 1.,  2.,  1.],
        [ 1.,  2.,  1.],
        [ 2.,  1.,  1.]]]))

"""




if __name__ == '__main__':
    node_instance = tutorial5_soccer()
    # node_instance.one_foot_stand()
    # node_instance.tutorial5_soccer_test()
    # node_instance.tutorial5_soccer_execute_test_by_tactile()
    # node_instance.stand()
    # node_instance.test()
    # node_instance.tutorial5_soccer_joint_test()
    node_instance.tutorial5_soccer_train()
    node_instance.tutorial5_soccer_test()
    # node_instance. tutorial5_soccer_execute_test_by_tactile()
    # node_instance.test()




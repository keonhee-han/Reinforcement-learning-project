from typing import Tuple
import pickle

from sensor_msgs.msg import Image, JointState
from naoqi_bridge_msgs.msg import HeadTouch
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np


class TrainingDataRecorder:

    def __init__(self):
        self.joint_angles = []
        self.joint_names = []
        self.blob_coordinates = (0, 0)
        self.tactile_button_pressed = False
        self.init_robot()
        rospy.Subscriber("joint_states", JointState, self.joint_states_callback)
        rospy.Subscriber("tactile_touch", HeadTouch, self.tactile_buttons_callback)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw", Image, self.camera_image_callback)

    def joint_states_callback(self, data):
        self.joint_names = data.name
        self.joint_angles = data.position

    def tactile_buttons_callback(self, data):
        rospy.loginfo("touch button: " + str(data.button) + " state: " + str(data.state))
        if data.button == 1 and data.state == 1:
            self.tactile_button_pressed = True

    def camera_image_callback(self, data):
        try:
            br = CvBridge()

            # Output debugging information to the terminal
            rospy.loginfo("receiving video frame")

            # Convert ROS Image message to OpenCV image
            current_frame = br.imgmsg_to_cv2(data)
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

            params = cv2.SimpleBlobDetector_Params()

            # Change thresholds
            params.minThreshold = 30
            params.maxThreshold = 255

            # Filter by Area.
            params.filterByArea = True
            params.minArea = 30

            # Filter by Circularity
            params.filterByCircularity = True
            params.minCircularity = 0.1

            params.filterByColor = True
            params.blobColor = 255

            lower_green = np.array([160, 100, 20])
            upper_green = np.array([179, 255, 255])
            # Threshold the HSV image to get only blue colors
            mask = cv2.inRange(current_frame, lower_green, upper_green)

            erode_kernel = np.ones((3, 3), np.uint8)
            eroded_img = cv2.erode(mask, erode_kernel, iterations=1)

            # dilate
            dilate_kernel = np.ones((10, 10), np.uint8)
            dilate_img = cv2.dilate(eroded_img, dilate_kernel, iterations=1)
            # Detect blobs.

            # Create a detector with the parameters
            detector = cv2.SimpleBlobDetector_create(params)

            # Detect blobs.
            keypoints = detector.detect(dilate_img)

            if not keypoints:
                return

            for blob in keypoints:
                max_size = 0
                x_coord = 0
                y_coord = 0
                maxObject = None
                for blob in keypoints:
                    if (blob.size > max_size):
                        max_size = blob.size
                        x_coord = blob.pt[0]
                        y_coord = blob.pt[1]
                        maxObject = blob

            if max_size == 0:
                return 0, 0

            # Round the coordinates to get pixel coordinates:
            x_pixel = round(x_coord)
            y_pixel = round(y_coord)

            # For better readability, round size to 3 decimal indices
            blob_size = round(max_size, 3)

            rospy.loginfo(
                "Biggest blob: x Coord: " + str(x_pixel) + " y Coord: " + str(y_pixel) + " Size: " + str(blobSize))

            # im_with_keypoints = cv2.drawKeypoints(current_frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.circle(frame, (int(kp_max.pt[0]), int(kp_max.pt[1])), int(kp_max.size), (0, 255, 0), 2)

            cv2.imshow("Keypoints", im_with_keypoints)

            cv2.waitKey(3)

            self.blob_coordinates = (x_pixel, y_pixel)

        except CvBridgeError as e:
            rospy.logerr(e)

    # Read RightShoulderPitch and RightShoulderRoll
    def read_joint_angles(self) -> Tuple[int, int]:

        right_shoulder_pitch = 0
        right_shoulder_roll = 0
        current_joint_angles = self.joint_angles
        for i in range(len(self.joint_names)):
            current_joint_name = self.joint_names[i]
            if current_joint_name == 'RightShoulderPitch':
                right_shoulder_pitch = self.joint_angles[i]
            elif current_joint_name == 'RightShoulderRoll':
                right_shoulder_roll == self.joint_angles[i]
        return right_shoulder_pitch, right_shoulder_roll

    # Init node and set default stiffness for elbow and head
    def init_robot(self):
        rospy.init_node('read_training_data', anonymous=True)
        # TODO: Set stiffness for shoulder and elbow


if __name__ == '__main__':

    training_data_recorder = TrainingDataRecorder()
    training_data_recorder.init_robot()

    # Record 150 training samples:
    # marker_position => input (x-data)
    # joint_angles_hip => output (y-data

    train_x = []
    train_y = []

    number_samples = 0
    while number_samples < 150:
        if training_data_recorder.tactile_button_is_pressed:
            marker_position = training_data_recorder.blob_coordinates
            if marker_position == (0, 0):
                continue
            joint_angles = training_data_recorder.read_joint_angles()
            train_x.append(marker_position)
            train_y.append(joint_angles)
            training_data_recorder.tactile_button_pressed = False
            number_samples += 1

    with open("train.pkl", "w") as f:
        pickle.dump([train_x, train_y], f)

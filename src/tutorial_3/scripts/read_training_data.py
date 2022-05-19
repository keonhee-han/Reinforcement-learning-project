from typing import Tuple
import pickle


# Read position of the marker using the camera of the robot and opencv
def read_marker_position() -> Tuple[int, int]:
    return 0, 0


# Read RightShoulderPitch and RightShoulderRoll
def read_joint_angles_hip() -> Tuple[int, int]:
    return 0, 0


# Return true if tacile button 1 is pressed in order to read in training data
def tactile_button_is_pressed() -> bool:
    return True


# Set default stiffness for elbow and head
def setup_robot():
    return


if __name__ == '__main__':

    setup_robot()

    # Record 150 training samples:
    # marker_position => input (x-data)
    # joint_angles_hip => output (y-data
    for i in range(150):
        if tactile_button_is_pressed():
            marker_position = read_marker_position()
            joint_angles_hip = read_joint_angles_hip()

    with open("train.pkl", "r") as f:
        train_x, train_y = pickle.load(f)

import numpy as np
import math

RESOLUTION_RECEPTIVE = 50
RECEPTIVE_FIELD_SIZE = 3
LEARNING_RATE = 0.1
TRAINING_ITERATIONS = 150


class CMACNetwork:
    def __int__(self):
        self.weights = np.full([50, 50, 2], 0.5)
        self.resolution_image = 200

    def train(self, data_samples):
        for sample in data_samples:
            training_input = sample[0]
            training_output = sample[1]
            self.training_step(training_input, training_output)

    def training_step(self, training_input, training_output):
        right_shoulder_pitch, right_shoulder_roll = training_output
        x_pixel, y_pixel = training_input

        # Convert to receptive field resolution
        x_pixel_converted = x_pixel / self.resolution_image * RESOLUTION_RECEPTIVE
        y_pixel_converted = y_pixel / self.resolution_image * RESOLUTION_RECEPTIVE

        # Identify boundaries of receptive field
        x_receptive_field_lower_boundary = 0
        if x_pixel_converted > RECEPTIVE_FIELD_SIZE:
            x_receptive_field_lower_boundary = math.ceil(x_pixel_converted - RECEPTIVE_FIELD_SIZE)

        x_receptive_field_upper_boundary = 50
        if x_pixel_converted > RESOLUTION_RECEPTIVE - RECEPTIVE_FIELD_SIZE:
            x_receptive_field_upper_boundary = math.ceil(x_pixel_converted + RECEPTIVE_FIELD_SIZE)

        y_receptive_field_lower_boundary = 0
        if y_pixel_converted > RECEPTIVE_FIELD_SIZE:
            y_receptive_field_lower_boundary = math.ceil(y_pixel_converted - RECEPTIVE_FIELD_SIZE)

        y_receptive_field_upper_boundary = 50
        if y_pixel_converted > RESOLUTION_RECEPTIVE - RECEPTIVE_FIELD_SIZE:
            y_receptive_field_upper_boundary = math.ceil(y_pixel_converted + RECEPTIVE_FIELD_SIZE)

        # Predict output by iterating over activated neurons
        right_shoulder_pitch_predicted = 0
        right_shoulder_roll_predicted = 0
        for i in range(x_receptive_field_lower_boundary, x_receptive_field_upper_boundary):
            for j in range(y_receptive_field_lower_boundary, y_receptive_field_upper_boundary):
                right_shoulder_pitch_predicted += self.weights[i][j][0]
                right_shoulder_roll_predicted += self.weights[i][j][1]

        # Compute error
        error_right_shoulder_pitch = right_shoulder_pitch - right_shoulder_pitch_predicted
        error_right_shoulder_roll = right_shoulder_roll - right_shoulder_roll_predicted

        # Update weights
        number_weights = RECEPTIVE_FIELD_SIZE * RECEPTIVE_FIELD_SIZE
        for i in range(x_receptive_field_lower_boundary, x_receptive_field_upper_boundary):
            for j in range(y_receptive_field_lower_boundary, y_receptive_field_upper_boundary):
                self.weights[i][j][0] += error_right_shoulder_pitch * LEARNING_RATE / number_weights
                self.weights[i][j][1] += error_right_shoulder_roll * LEARNING_RATE / number_weights


if __name__ == '__main__':
    cmac = CMACNetwork()

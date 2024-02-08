import numpy as np
from scipy.spatial.transform import Rotation

class Actor:
    def __init__(self, pos, orientation, vel, precisions, name):
        self.position = pos
        self.orientation = orientation
        self.velocity = vel

        self.prev_measured_linear_accel = np.array([0.0, 0.0, 0.0])
        self.prev_measured_angular_vel = np.array([0.0, 0.0, 0.0])

        self.precisions = precisions
        self.name = name

        # for preintegration
        self.prev_position = pos
        self.prev_orientation = orientation

        # for PID
        self.prev_pos_error = np.array([0.0, 0.0, 0.0])
        self.sum_pos_error = np.array([0.0, 0.0, 0.0])

    def integrate_imu_paper(self, measured_linear_accel, measured_angular_vel, dt):
        """Integrate IMU measurements, based on "On-Manifold Preintegration for Real-Time
            Visual-Inertial Odometry"

        Args:
            measured_linear_accel (np.ndarray): Measured linear acceleration in body frame in m/s^2.
            measured_angular_vel (np.ndarray): Measured angular velocity in body frame in rad/s.
            dt (float): Time step.
        """
        prev_velocity = self.velocity
        prev_orientation = self.orientation
        gravity = np.array([0.0, 0.0, -9.8])
        if np.linalg.norm(self.prev_measured_angular_vel* dt) == 0.0:
            change_in_orientation = Rotation.identity()
        else:
            change_in_orientation = lie_exp(self.prev_measured_angular_vel * dt)

        self.orientation = prev_orientation * change_in_orientation

        self.velocity = prev_velocity + (gravity * dt) + prev_orientation.apply(self.prev_measured_linear_accel * dt)

        change_in_position = (prev_velocity * dt) + (0.5 * gravity * dt * dt) + (0.5 * prev_orientation.apply(self.prev_measured_linear_accel) * dt * dt)

        self.position = self.position + change_in_position

        self.prev_measured_linear_accel = measured_linear_accel
        self.prev_measured_angular_vel = measured_angular_vel

        return prev_orientation.inv().apply(change_in_position), change_in_orientation

def lie_exp(vector):
    skew_matrix = hat_operator(vector)
    magnitude = np.linalg.norm(vector)

    result = Rotation.identity().as_matrix() + ((np.sin(magnitude) / magnitude) * skew_matrix) + (((1 - np.cos(magnitude)) / (magnitude * magnitude)) * (np.dot(skew_matrix, skew_matrix)))
    return Rotation.from_matrix(result)

    # this is a good approximation
    # return Rotation.from_matrix(skew_matrix + Rotation.identity().as_matrix())

def hat_operator(vector):
    return np.array([[0.0, -vector[2], vector[1]],
                     [vector[2], 0.0, -vector[0]],
                     [-vector[1], vector[0], 0.0]])
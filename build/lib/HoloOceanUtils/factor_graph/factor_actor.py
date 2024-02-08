import numpy as np
import math as math
from scipy.spatial.transform import Rotation
from HoloOceanUtils.factor_graph.actor import Actor
import itertools

from py_factor_graph.variables import PoseVariable3D, LandmarkVariable3D
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.measurements import FGRangeMeasurement, PoseMeasurement3D, FGBearingMeasurement
from py_factor_graph.io.pyfg_text import save_to_pyfg_text
from py_factor_graph.utils.matrix_utils import get_measurement_precisions_from_covariances
from py_factor_graph.utils.name_utils import get_robot_idx_from_char

def generate_landmarks_uniform(count, bounds):
    landmarks = []
    for i in range(count):
        landmark = LandmarkVariable3D("L" + str(i), (np.random.uniform(bounds[0], bounds[1]), np.random.uniform(bounds[2], bounds[3]), np.random.uniform(bounds[4], bounds[5])))
        landmarks.append(landmark)
    return landmarks

def get_variances_imu(sensors):
    for sensor in sensors:
        if sensor["sensor_type"] == "IMUSensor":
            return (sensor["configuration"]["AccelSigma"] ** 2.0, sensor["configuration"]["AngVelSigma"] ** 2.0)

class FactorGraphCollector:
    def __init__(self, env, parameters, index):
        self.pyfg = FactorGraphData(dimension=3)
        self.counter = 0
        self.done_collecting = False

        # set up actors
        self.actors = []
        for agent in env._scenario["agents"]:
            name = agent["agent_name"]
            if "location" in agent:
                init_pos = agent["location"]
            else:
                init_pos = [0.0, 0.0, 0.0]

            init_rotation = [0.0, 0.0, 0.0]
            init_velocity = [0.0, 0.0, 0.0]

            (variance_lin, variance_ang) = get_variances_imu(agent["sensors"])

            # variances are multiplied by the number of time steps in a capture, this is an approximation
            n = int(env._ticks_per_sec * parameters["capture_length"] * parameters["num_captures"])
            precisions = get_measurement_precisions_from_covariances(n * variance_lin, n * variance_ang, 6)

            self.actors.append(Actor(np.array(init_pos), Rotation.from_euler('xyz', init_rotation), np.array(init_velocity), precisions, name))

        # set up landmarks
        self.landmarks = generate_landmarks_uniform(parameters["num_landmarks"], parameters["landmark_bounds"])
        for landmark in self.landmarks:
            self.pyfg.add_landmark_variable(landmark)

        self.parameters = parameters
        self.index = index

    def update_actors(self, state, dt):
        for actor in self.actors:
            actor.integrate_imu_paper(state[actor.name]['IMUSensor'][0], state[actor.name]['IMUSensor'][1], dt)

    def update_graph(self, state, now):
        if self.counter == 0:
            for actor in self.actors:
                actor.prev_measured_linear_accel = state[actor.name]['IMUSensor'][0]
                actor.prev_measured_angular_vel = state[actor.name]['IMUSensor'][1]


        try:
            print(state['A']['AcousticBeaconSensor'])
            print(state['A']['LocationSensor'])

        except:
            # pass
            print(state['L']['LocationSensor'])
            print("Beacon doesn't exist")


        self.add_ground_truth_poses(state, now)
        self.add_landmark_ranges(state, now)
        self.add_actor_ranges(state, now)
        self.add_actor_bearings(state, now)
        if self.counter != 0:
            self.add_odometry_pre_integrate(now)
        self.counter += 1

        print("trial", str(self.index) + ":", round((float(self.counter) / float(self.parameters["num_captures"])) * 100.0, 1), r"% complete")
        # self.pyfg.print_summary()

        # save factor graphs
        if self.counter >= self.parameters["num_captures"] and not self.done_collecting:
            save_name = self.parameters["pyfg_save_location"] + self.parameters["pyfg_save_name"]+ "_" + str(self.index) + ".pyfg"
            save_to_pyfg_text(self.pyfg, save_name)
            self.done_collecting = True

    def add_ground_truth_poses(self, state, now):
        for actor in self.actors:
            pose = PoseVariable3D(actor.name + str(self.counter), tuple(np.array(state[actor.name]['LocationSensor'], dtype=float)), Rotation.from_euler('xyz', state[actor.name]['RotationSensor'], degrees=True).as_matrix(), now)
            self.pyfg.add_pose_variable(pose)

    def add_landmark_ranges(self, state, now):
        for landmark in self.landmarks:
            for actor in self.actors:
                # TODO: This section uses the location sensor, not the acoustic modem that the actor_ranges measuremetns use
                landmark_range = float(np.linalg.norm(state[actor.name]['LocationSensor'] - list(landmark.true_position)) + np.random.normal(0.0, self.parameters["landmark_range_sigma"]))
                landmark_range_measurement = FGRangeMeasurement((actor.name + str(self.counter), landmark.name), landmark_range, self.parameters["landmark_range_sigma"], now)
                self.pyfg.add_range_measurement(landmark_range_measurement)

    def add_actor_ranges(self, state, now):
        for actor1, actor2 in itertools.combinations(self.actors, 2):
            actor_range = float(state['A']['AcousticBeaconSensor'][5]) + np.random.normal(0.0, self.parameters["auv_range_sigma"])
            actor_range_measurement = FGRangeMeasurement((actor1.name + str(self.counter), actor2.name + str(self.counter)), actor_range, self.parameters["auv_range_sigma"], now)
            self.pyfg.add_range_measurement(actor_range_measurement)

    def add_actor_bearings(self, state, now):
        for actor1, actor2 in itertools.combinations(self.actors, 2):
            actor_bearing_gt_elevation = state['A']['AcousticBeaconSensor'][4]
            actor_bearing_gt_azimuth = state['A']['AcousticBeaconSensor'][3]
            # Elevation and Azimuth sigmas vary w.r.t elevation due to 4-element USBL beam pattern
            # Designed to produce similar results to those found in "Performance of a Low-Power One-Way Travel-Time Inverted Ultra-Short Baseline Navigation System"- Mike Jakuba, 2019
            # Calculate Elevation Sigma
            elev_sigma_scale = 0.35
            elev_sigma_slope = -2.5
            actor_bearing_elevation_sigma = elev_sigma_scale*math.exp(elev_sigma_slope*actor_bearing_gt_elevation)
            # Calculate Azimuth Sigma
            az_sigma_scale = 0.00218
            az_sigma_slope = 2.2
            az_sigma_offset = 0.00654
            actor_bearing_azimuth_sigma = az_sigma_scale*math.exp(az_sigma_slope*actor_bearing_gt_elevation) + az_sigma_offset
            # Calculate bearing measurements
            actor_bearing_elevation = actor_bearing_gt_elevation + np.random.normal(0.0, actor_bearing_elevation_sigma)
            actor_bearing_azimuth = actor_bearing_gt_azimuth + np.random.normal(0.0, actor_bearing_azimuth_sigma)
            actor_bearing_measurement = FGBearingMeasurement((actor1.name + str(self.counter), actor2.name + str(self.counter)), actor_bearing_azimuth, actor_bearing_elevation, actor_bearing_azimuth_sigma, actor_bearing_elevation_sigma, now)
            self.pyfg.add_bearing_measurement(actor_bearing_measurement)

    def add_odometry_pre_integrate(self, now):
        for actor in self.actors:
            d_position_world = actor.position - actor.prev_position
            d_position_actor = actor.prev_orientation.inv().apply(d_position_world)
            d_rotation_actor = actor.prev_orientation.inv() * actor.orientation

            pose_measurement = PoseMeasurement3D(actor.name + str(self.counter - 1), actor.name + str(self.counter), d_position_actor, d_rotation_actor.as_matrix(), actor.precisions[0], actor.precisions[1], now)

            self.pyfg.add_odom_measurement(get_robot_idx_from_char(actor.name), pose_measurement)

            actor.prev_position = actor.position
            actor.prev_orientation = actor.orientation
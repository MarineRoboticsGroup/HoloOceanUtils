import numpy as np
from scipy.spatial.transform import Rotation
from HoloOceanUtils.factor_graph.actor import Actor
import itertools

from py_factor_graph.variables import PoseVariable3D, LandmarkVariable3D
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.measurements import FGRangeMeasurement, PoseMeasurement3D
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

        self.add_ground_truth_poses(state, now)
        self.add_landmark_ranges(state, now)
        self.add_actor_ranges(state, now)
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
                landmark_range = float(np.linalg.norm(state[actor.name]['LocationSensor'] - list(landmark.true_position)) + np.random.normal(0.0, self.parameters["landmark_range_sigma"]))
                landmark_range_measurement = FGRangeMeasurement((actor.name + str(self.counter), landmark.name), landmark_range, self.parameters["landmark_range_sigma"], now)
                self.pyfg.add_range_measurement(landmark_range_measurement)

    def add_actor_ranges(self, state, now):
        for actor1, actor2 in itertools.combinations(self.actors, 2):
            actor_range = float(np.linalg.norm(state[actor1.name]['LocationSensor'] - state[actor2.name]['LocationSensor']) + np.random.normal(0.0, self.parameters["auv_range_sigma"]))
            actor_range_measurement = FGRangeMeasurement((actor1.name + str(self.counter), actor2.name + str(self.counter)), actor_range, self.parameters["auv_range_sigma"], now)
            self.pyfg.add_range_measurement(actor_range_measurement)

    def add_odometry_pre_integrate(self, now):
        for actor in self.actors:
            d_position_world = actor.position - actor.prev_position
            d_position_actor = actor.prev_orientation.inv().apply(d_position_world)
            d_rotation_actor = actor.prev_orientation.inv() * actor.orientation

            pose_measurement = PoseMeasurement3D(actor.name + str(self.counter - 1), actor.name + str(self.counter), d_position_actor, d_rotation_actor.as_matrix(), actor.precisions[0], actor.precisions[1], now)

            self.pyfg.add_odom_measurement(get_robot_idx_from_char(actor.name), pose_measurement)

            actor.prev_position = actor.position
            actor.prev_orientation = actor.orientation
"""
class FactorActorIMU:
    def __init__(self, env, parameters, index):
        self.pyfg = FactorGraphData(dimension=3)
        self.counter = 0
        self.done_collecting = False

        for agent in env._scenario["agents"]:
            if agent["agent_name"] == "auv0":
                auv0_init_pos = agent["location"]
                (auv0_variance_lin, auv0_variance_ang) = get_variances_imu(agent["sensors"])
            elif agent["agent_name"] == "diver":
                diver_init_pos = agent["location"]
                (diver_variance_lin, diver_variance_ang) = get_variances_imu(agent["sensors"])


        self.auv0_actor = Actor(np.array(auv0_init_pos), Rotation.from_euler('xyz', [0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        self.prev_auv0_actor = copy.deepcopy(self.auv0_actor)

        self.diver_actor = Actor(np.array(diver_init_pos), Rotation.from_euler('xyz', [0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        self.prev_diver_actor = copy.deepcopy(self.diver_actor)

        n = int(env._ticks_per_sec * parameters["capture_length"] * parameters["num_captures"])
        self.auv_precision = get_measurement_precisions_from_covariances(n * auv0_variance_lin,  n * auv0_variance_ang, 6)
        self.diver_precision = get_measurement_precisions_from_covariances(n * diver_variance_lin,  n * diver_variance_ang, 6)

        self.range_sigma = parameters["range_sigma"]

        self.landmarks = generate_landmarks_uniform(parameters["num_landmarks"], parameters["landmark_bounds"][0], parameters["landmark_bounds"][1], parameters["landmark_bounds"][2], parameters["landmark_bounds"][3], parameters["landmark_bounds"][4], parameters["landmark_bounds"][5])
        for landmark in self.landmarks:
            self.pyfg.add_landmark_variable(landmark)

    def update(self, state, num_captures, dt, now):
        if self.counter == 0:
            self.auv0_actor.prev_measured_linear_accel = state["auv0"]['IMUSensor'][0]
            self.auv0_actor.prev_measured_angular_vel = state["auv0"]['IMUSensor'][1]

            self.diver_actor.prev_measured_linear_accel = state["diver"]['IMUSensor'][0]
            self.diver_actor.prev_measured_angular_vel = state["diver"]['IMUSensor'][1]

        self.update_actors(state, dt)

        if 'AcousticBeaconSensor' in state["auv0"]:
            print("trial ", (float(self.counter) / float(num_captures)) * 100.0, r"% complete")
            self.add_poses(state, now)
            self.add_landmark_ranges(state, now)
            self.add_diver_auv_range(state, now)
            if self.counter != 0:
                self.add_odometry_pre_integrate(now)
            self.counter += 1



        # save factor graphs
        if self.counter >= num_captures and not self.done_collecting:
            save_name = self.parameters["pyfg_save_location"] + self.parameters["pyfg_save_name"]+ "_" + str(self.index) + ".pyfg"
            save_to_pyfg_text(self.pyfg, save_name)
            self.done_collecting = True

    def add_landmark_ranges(self, state, now):
        for landmark in self.landmarks:
            auv0_landmark_range = float(np.linalg.norm(state["auv0"]['LocationSensor'] - list(landmark.true_position)) + np.random.normal(0.0, self.range_sigma))
            auv0_landmark_range_measurement = FGRangeMeasurement(("A" + str(self.counter), landmark.name), auv0_landmark_range, self.range_sigma, now)
            self.pyfg.add_range_measurement(auv0_landmark_range_measurement)

            diver_landmark_range = float(np.linalg.norm(state["diver"]['LocationSensor'] - list(landmark.true_position)) + np.random.normal(0.0, self.range_sigma))
            diver_landmark_range_measurement = FGRangeMeasurement(("B" + str(self.counter), landmark.name), diver_landmark_range, self.range_sigma, now)
            self.pyfg.add_range_measurement(diver_landmark_range_measurement)

    def add_poses(self, state, now):
        auv0_pose = PoseVariable3D("A" + str(self.counter), tuple(np.array(state["auv0"]['LocationSensor'], dtype=float)), Rotation.from_euler('xyz', state["auv0"]['RotationSensor'], degrees=True).as_matrix(), now)
        diver_pose = PoseVariable3D("B" + str(self.counter), tuple(np.array(state["diver"]['LocationSensor'], dtype=float)), Rotation.from_euler('xyz', state["diver"]['RotationSensor'], degrees=True).as_matrix(), now)

        self.pyfg.add_pose_variable(auv0_pose)
        self.pyfg.add_pose_variable(diver_pose)

    def add_diver_auv_range(self, state, now):
        diver_auv0_range = float(state["auv0"]['AcousticBeaconSensor'][5] + np.random.normal(0.0, self.range_sigma))
        #perfect_range = float(np.linalg.norm(self.auv0_cur_position - self.diver_cur_position))
        range_measurement = FGRangeMeasurement(("A" + str(self.counter), "B" + str(self.counter)), diver_auv0_range, self.range_sigma, now)
        self.pyfg.add_range_measurement(range_measurement)

    def update_actors(self, state, dt):
        self.auv0_actor.integrate_imu_paper(state["auv0"]['IMUSensor'][0], state["auv0"]['IMUSensor'][1], dt)
        self.diver_actor.integrate_imu_paper(state["diver"]['IMUSensor'][0], state["diver"]['IMUSensor'][1], dt)

    def add_odometry_pre_integrate(self, now):
        auv0_d_position_world = self.auv0_actor.position - self.prev_auv0_actor.position
        auv0_d_position = self.prev_auv0_actor.orientation.inv().apply(auv0_d_position_world)
        auv0_d_rotation = self.prev_auv0_actor.orientation.inv() * self.auv0_actor.orientation

        diver_d_position_world = self.diver_actor.position - self.prev_diver_actor.position
        diver_d_position = self.prev_diver_actor.orientation.inv().apply(diver_d_position_world)
        diver_d_rotation = self.prev_diver_actor.orientation.inv() * self.diver_actor.orientation


        auv0_pose_measurement = PoseMeasurement3D("A" + str(self.counter - 1), "A" + str(self.counter), auv0_d_position, auv0_d_rotation.as_matrix(), self.auv_precision[0], self.auv_precision[1], now)
        diver_pose_measurement = PoseMeasurement3D("B" + str(self.counter - 1), "B" + str(self.counter), diver_d_position, diver_d_rotation.as_matrix(), self.diver_precision[0], self.diver_precision[1], now)

        self.pyfg.add_odom_measurement(0, auv0_pose_measurement)
        self.pyfg.add_odom_measurement(1, diver_pose_measurement)

        self.prev_auv0_actor = copy.deepcopy(self.auv0_actor)
        self.prev_diver_actor = copy.deepcopy(self.diver_actor)

class FactorActorPerfect:
    def __init__(self):
        self.pyfg = FactorGraphData(dimension=3)
        self.counter = 0

        self.auv0_prev_orientation_world = Rotation.from_euler('xyz', [0.0, 0.0, 0.0])
        self.diver_prev_orientation_world = Rotation.from_euler('xyz', [0.0, 0.0, 0.0])
        self.auv0_cur_orientation_world = Rotation.from_euler('xyz', [0.0, 0.0, 0.0])
        self.diver_cur_orientation_world = Rotation.from_euler('xyz', [0.0, 0.0, 0.0])

        self.auv0_cur_position = np.array([5.0, 0.0, -10.0])
        self.diver_cur_position = np.array([0.0, 0.0, -10.0])
        self.auv0_prev_pos = np.array([5.0, 0.0, -10.0])
        self.diver_prev_pos = np.array([0.0, 0.0, -10.0])

        self.prev_time = time.time()

    def update(self, state):
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        self.auv0_cur_position = np.array(state["auv0"]['LocationSensor'], dtype=float)
        self.diver_cur_position = np.array(state["diver"]['LocationSensor'], dtype=float)

        self.auv0_cur_orientation_world = Rotation.from_euler('xyz', state["auv0"]['RotationSensor'], degrees=True)
        self.diver_cur_orientation_world = Rotation.from_euler('xyz', state["diver"]['RotationSensor'], degrees=True)

        self.add_poses(now)

        self.add_range(now)

        # add odometry measurements
        if self.counter != 0:
            self.add_odometry(now)

        # update previous values
        self.auv0_prev_pos = self.auv0_cur_position
        self.diver_prev_pos = self.diver_cur_position
        self.auv0_prev_orientation_world = self.auv0_cur_orientation_world
        self.diver_prev_orientation_world = self.diver_cur_orientation_world

        # save factor graph
        if self.counter == 500:
            save_to_pyfg_text(self.pyfg, "diver_auv_perfect_with_current.pyfg")
            print("saved factor graph")

        self.counter += 1

    def add_poses(self, now):
        auv0_pose = PoseVariable3D("A" + str(self.counter), tuple(self.auv0_cur_position), self.auv0_cur_orientation_world.as_matrix(), now)
        diver_pose = PoseVariable3D("B" + str(self.counter), tuple(self.diver_cur_position), self.diver_cur_orientation_world.as_matrix(), now)

        self.pyfg.add_pose_variable(auv0_pose)
        self.pyfg.add_pose_variable(diver_pose)

    def add_range(self, now):
        #diver_auv0_range = float(state["auv0"]['AcousticBeaconSensor'][5])
        perfect_range = float(np.linalg.norm(self.auv0_cur_position - self.diver_cur_position))
        range_measurement = FGRangeMeasurement(("A" + str(self.counter), "B" + str(self.counter)), perfect_range, 1.0, now)
        self.pyfg.add_range_measurement(range_measurement)

    def add_odometry(self, now):
        auv0_d_position_world = self.auv0_cur_position - self.auv0_prev_pos
        auv0_d_position = self.auv0_prev_orientation_world.inv().apply(auv0_d_position_world)

        diver_d_position_world = self.diver_cur_position - self.diver_prev_pos
        diver_d_position = self.diver_prev_orientation_world.inv().apply(diver_d_position_world)

        auv0_d_rotation = self.auv0_prev_orientation_world.inv() * self.auv0_cur_orientation_world
        diver_d_rotation = self.diver_prev_orientation_world.inv() * self.diver_cur_orientation_world

        auv0_pose_measurement = PoseMeasurement3D("A" + str(self.counter - 1), "A" + str(self.counter), auv0_d_position, auv0_d_rotation.as_matrix(), 1.0, 1.0, now)
        diver_pose_measurement = PoseMeasurement3D("B" + str(self.counter - 1), "B" + str(self.counter), diver_d_position, diver_d_rotation.as_matrix(), 1.0, 1.0, now)

        self.pyfg.add_odom_measurement(0, auv0_pose_measurement)
        self.pyfg.add_odom_measurement(1, diver_pose_measurement)
"""
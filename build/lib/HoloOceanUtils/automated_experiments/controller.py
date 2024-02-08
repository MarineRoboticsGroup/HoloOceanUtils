from os import getxattr
from pynput import keyboard
import numpy as np

class UserController:
    def __init__(self):
        self.pressed_keys = list()
        self.listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        if hasattr(key, 'char'):
            self.pressed_keys.append(key.char)
            self.pressed_keys = list(set(self.pressed_keys))

    def on_release(self, key):
        if hasattr(key, 'char'):
            self.pressed_keys.remove(key.char)

    def parse_keys(self, val):
        command = np.zeros(8)
        if 'i' in self.pressed_keys:
            command[0:4] += val
        if 'k' in self.pressed_keys:
            command[0:4] -= val
        if 'f' in self.pressed_keys:
            command[[4,7]] += val / 2
            command[[5,6]] -= val / 2
        if 'h' in self.pressed_keys:
            command[[4,7]] -= val / 2
            command[[5,6]] += val / 2

        if 't' in self.pressed_keys:
            command[4:8] += val
        if 'g' in self.pressed_keys:
            command[4:8] -= val
        if 'j' in self.pressed_keys:
            command[[4,6]] += val
            command[[5,7]] -= val
        if 'l' in self.pressed_keys:
            command[[4,6]] -= val
            command[[5,7]] += val
        return command

# TODO: Open loop desired position functions
def desired_pos_direct(start, counter, duration, origin, destination):
    """Generates a set of agent goal positions along a linear trajectory at 300Hz
    Args:
        start: The trajectory start time (in counts)
        counter: The current scenario counter value
        duration: The trajectory duration (in captures)
        origin: The start point [x,y,z]
        destination: The end point [x,y,z]
    Output:
        goal_pos: A goal position at 300Hz
    """
    path_ticks = 300*duration
    portion = (counter-start) / path_ticks
    path_vector = np.array(destination) - np.array(origin)
    progress = path_vector * portion
    goal_pos = np.array(origin) + progress
    return goal_pos    
def desired_pos_circling(start_time, t, period, radius, reference_obj):
    """Generates a set of goal positions that circle a reference point
    Args:
    Output:
        goal_pos: A goal position at 300Hz
    """
    portion = ((t - start_time) % period) / period
    goal_pos_diver = np.array([-radius, 0.0, 0.0])
    theta = 2.0 * np.pi * portion
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    path_vector = np.dot(rotation_matrix, goal_pos_diver)
    goal_pos = np.array(reference_obj) + path_vector
    return goal_pos

# TODO: Closed loop desired position functions
def desired_pos_Hz(input, basis, input_rate, tick_rate, speed):
    # Keep this for use with closed loop controller
    """Generates goal position at 300Hz using 1Hz nav sensor input
    Args:
        input: The sensor used for navigation (at 1Hz)
        basis: The LocationSensor (at 300Hz) (needed to provide world-frame positions)
        input_rate: Typically 1Hz (but can be any slower frequency in 1Hz increments)
        tick_rate: 300Hz (limited by controller)
        speed: Agent speed
    """
    # Develop Goal position at 1Hz
    nav_goal = [30, 0, -20] # from behavior controller
    # Calculate LF step vector (how far do you go in 1sec)
    disp = np.array(nav_goal) - np.array(input)
    #print("disp: ", disp)
    lf_unit = disp / np.linalg.norm(disp)
    #print("direction: ", lf_unit)
    step = speed / input_rate
    lf_vector = np.array(lf_unit) * step
    #print("lf_vect: ", lf_vector)
    # Convert to HF step vector (how far do you go in 0.0033sec)
    hf_vector = lf_vector / tick_rate
    # Where is this in the world frame
    goal_pos = lf_vector + basis
    #print("goal_pos: ", goal_pos
    #print("goal_pos: ", goal_pos)
    return goal_pos
def desired_pos_inferred(parameters, gtsam_pose, dt_sim):
    # Keep for use with closed loop controller
    origin = [0, 0, -20]
    destination = [30, 0 , -20]
    basis = state["B"]["LocationSensor"]
    speed = 0.5
    lf_vector = desired_pos_simple(origin, destination, 1, speed)
    lf_goal = basis + vector
    return goal_postion #based on a gtsam-based update

    #print("gtsampose: ", gtsam_pose)
    if gtsam_pose == []:
        pass
    else:
        gtsam_trans_x = gtsam_pose[0][3]
        gtsam_trans_y = gtsam_pose[1][3]
        gtsam_trans_z = gtsam_pose[2][3]
        gtsam_translation = [gtsam_trans_x, gtsam_trans_y, gtsam_trans_z]
    #print("Controller Translation: ", gtsam_translation)
    goal_pos = desired_pos_vector(gtsam_translation, parameters["diver_destination"], parameters["diver_speed"], dt_sim)
    return goal_pos

    # gtsam translation is in the same format as goal pos

# TODO: Revise loop feedback
    # Use existing "open-loop" controller for the AUV
    # Develop a "closed-loop" controller for use with the diver
def auv_command(parameters, state, prev_error, sum_error, dt_sim, counter, start_time, now_sim):
    # Trajectory Planning
    #if counter < 9000:
    # Direct trajectory:
    #goal_pos = desired_pos_direct(0, counter, parameters["num_captures"], [0, 20, -30], [30, 20, -30])
    #else:
        #goal_pos = desired_pos_direct(0, counter, parameters["num_captures"], [30, 20, -30], [0, 0, -30])

    # Circling trajectory:
    goal_pos = desired_pos_circling(start_time, now_sim, parameters["circling_period"], parameters["circling_radius"], state["B"]['LocationSensor'])
    return goal_pos

# TODO: Torpedo AUV control function
def tauv_command(parameters, state, prev_error, sum_error, dt_sim, counter, start_time, now_sim):
#     # Trajectory Planning

#     # Circling trajectory:
#     goal_pos = desired_pos_circling(start_time, now_sim, parameters["circling_period"], parameters["circling_radius"], state["B"]['LocationSensor'])
#     goal_pos_tauv = (goal_pos - np.array(state["A"]['LocationSensor']))

#     # Mode 0 (Fins and Thrust)
    

#     # Controller Errors
#     sum_error += goal_pos_auv * dt_sim

#     d_error = (goal_pos_auv - prev_error) / dt_sim
#     prev_error = goal_pos_auv

#     auv_command = k_p * goal_pos_auv + k_d * d_error + k_i * sum_error

    return tauv_command

def diver_command_open(parameters, state, prev_error, sum_error, dt_sim, counter, start_time, now_sim, gtsam_pose):
    # Compares goal_pos and location sensor to create command inputs
    # Navigates perfectly, good for testing
    # Trajectory Planning
    # Direct trajectory
    goal_pos = desired_pos_direct(0, counter, parameters["num_captures"], parameters["diver_origin"], parameters["diver_destination"])
    # Circling trajectory
    #goal_pos = desired_pos_circling(start_time, now_sim, parameters["circling_period"], parameters["circling_radius"], [0, 0, -60])
    return goal_pos

# TODO: Provide navigation feedback at 1Hz from GTSAM or odometry
    # Goal is to provide realistic navigation
def diver_command_closed(parameters, state, prev_error, sum_error, dt_sim, counter, start_time, now_sim, gtsam_pose):
    # Trajectory Planning
    # Direct trajectory
    goal_pos = desired_pos_direct(0, counter, parameters["num_captures"], parameters["diver_origin"], parameters["diver_destination"])
    # Circling trajectory
    #goal_pos = desired_pos_circling(start_time, now_sim, parameters["circling_period"], parameters["circling_radius"], [0, 0, -60])
    return goal_pos

def choose_random_commands(force, random_generator):
    commands = set()
    x_commands = [None, 'forward_fast', 'forward_slow']
    z_commands = [None, None, 'up', 'down']
    yaw_commands = [None, None,'turn_left', 'turn_right']

    commands.add(random_generator.choice(x_commands))
    commands.add(random_generator.choice(z_commands))
    commands.add(random_generator.choice(yaw_commands))
    return parse_commands(commands, force)

def parse_commands(commands, force):
    command = np.zeros(8)
    if 'up' in commands:
        command[0:4] += force / 2
    if 'down' in commands:
        command[0:4] -= force / 2
    if 'turn_left' in commands:
        command[[4,7]] += force / 8
        command[[5,6]] -= force / 8
    if 'turn_right' in commands:
        command[[4,7]] -= force / 8
        command[[5,6]] += force / 8

    if 'forward_fast' in commands:
        command[4:8] += force
    if 'forward_slow' in commands:
        command[4:8] += force / 2

    return command

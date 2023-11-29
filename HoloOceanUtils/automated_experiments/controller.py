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

def desired_pos_circling(start_time, t, period, radius):
    portion = ((t - start_time) % period) / period
    #print("portion: ", portion)
    goal_pos_diver = np.array([radius, 0.0, 0.0])
    theta = 2.0 * np.pi * portion
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    return np.dot(rotation_matrix, goal_pos_diver)

def auv_command(parameters, state, prev_error, sum_error, dt_sim, start_time, now_sim):
    # PID controller gains
    k_p = 0.3
    k_d = 0.2
    k_i = 0.0

    # calculare goal pos in diver frame
    goal_pos_diver = desired_pos_circling(start_time, now_sim, parameters["circling_period"], parameters["circling_radius"])

    # calculate goal pos in world frame
    # goal_pos_world = np.array(state["B"]['LocationSensor']) + goal_pos_diver

    # calculate goal pos in world frame with a safety offset to the diver [TODO] This causes errors with control scheme 2.
    goal_pos_world = np.array(state["B"]['LocationSensor']) + goal_pos_diver
    goal_pos_world[2] = goal_pos_world[2] + parameters["diver_depth_offset"]

    # calculate goal pos in auv frame (the error of our PID controller)
    goal_pos_auv = (goal_pos_world - np.array(state["A"]['LocationSensor']))

    sum_error += goal_pos_auv * dt_sim

    d_error = (goal_pos_auv - prev_error) / dt_sim
    prev_error = goal_pos_auv

    auv_command = k_p * goal_pos_auv + k_d * d_error + k_i * sum_error
    return auv_command

def choose_random_commands(force, random_generator):
    commands = set()
    x_commands = [None, 'forward_fast', 'forward_slow']
    # z_commands = [None, None, 'up', 'down']
    # changed to keep diver at the constant depth
    z_commands = [None, None, None, None]
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

from holoocean_ros_wrapper import ROSWrapper
import holoocean
from holoocean.environments import HoloOceanEnvironment

import rospy
import numpy as np
import message_filters
import math

import std_msgs.msg
import sensor_msgs.msg
import geometry_msgs.msg
from holo_wrapper.msg import AcousticBeaconSensor
from actor import Actor
from controller import Controller
from scipy.spatial.transform import Rotation


def imu_callback(msg, actor):
    now = rospy.get_time()
    dt = now - actor.prev_time
    actor.prev_time = now

    lin_accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
    ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

    actor.integrate_imu(lin_accel, ang_vel, dt)

def imu_callback_test(msg, actors):
    now = rospy.get_time()
    dt = now - actors[0].prev_time
    actors[0].prev_time = now

    lin_accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
    ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
    for index, actor in enumerate(actors):
        portion = (index + 1) / len(actors)
        #print("portion: ", portion)
        actor.integrate_imu(lin_accel, ang_vel, dt * portion)

""" def pos_callback(diver_pos_msg, auv0_pos_msg):
    #print("diver_pos_msg: ", diver_pos_msg)
    global diver_pos, auv0_pos
    diver_pos = np.array([diver_pos_msg.point.x, diver_pos_msg.point.y, diver_pos_msg.point.z])
    auv0_pos = np.array([auv0_pos_msg.point.x, auv0_pos_msg.point.y, auv0_pos_msg.point.z]) """
rospy.init_node("wrapper_node", anonymous=True)


def desired_pos(start_time, t, period):
    portion = ((t - start_time) % period) / period
    #print("portion: ", portion)
    goal_pos_diver = np.array([5.0, 0.0, 0.0])
    theta = 2.0 * math.pi * portion
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    return np.dot(rotation_matrix, goal_pos_diver)

config = {
    "name": "Hovering",
    "world": "sailingpavilion",
    "main_agent": "diver",
    "agents": [
            {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "LocationSensor"
                },
                {
                    "sensor_type": "AcousticBeaconSensor",
                    "location": [0,0,0],
                    "configuration": {
                        "id": 0
                    }
                },
                {
                    "sensor_type": "IMUSensor",
                    "configuration": {
                        "ReturnBias": True
                    }
                }
            ],
            "control_scheme": 1,
            "location": [0, 3, -10]
            },
            {
            "agent_name": "diver",
            "agent_type": "ScubaDiver",
            "sensors": [
                {
                    "sensor_type": "LocationSensor"
                },
                {
                    "sensor_type": "AcousticBeaconSensor",
                    "location": [0,0,0],
                    "configuration": {
                        "id": 1
                    }
                },
                {
                    "sensor_type": "CollisionSensor"
                },
                {
                    "sensor_type": "IMUSensor",
                    "configuration": {
                        "ReturnBias": True
                    }
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -10]
            }
        ],

    "window_width":  1280,
    "window_height": 720
}

if __name__ == "__main__":
    #with HoloOceanEnvironment(scenario=config, start_world=False) as env:
        k_p = 0.3
        k_d = 0.2
        k_i = 0.0
        prev_error = [0.0, 0.0, 0.0]
        sum_error = [0.0, 0.0, 0.0]
        prev_time = rospy.get_time()


        controller = Controller()

        env = holoocean.make("sailingpavilion-Hovering")
        use_current_ptr = env._client.malloc("use_current", [1], np.bool_)
        use_current_ptr[0] = False

        env.reset()

        ros_wrapper = ROSWrapper(env)

        # diver_pos_sub = message_filters.Subscriber("diver_LocationSensor", geometry_msgs.msg.PointStamped)
        # auv0_pos_sub = message_filters.Subscriber("auv0_LocationSensor", geometry_msgs.msg.PointStamped)

        # ts = message_filters.TimeSynchronizer([diver_pos_sub, auv0_pos_sub], 10)
        # ts.registerCallback(pos_callback)

        auv0_command_pub = rospy.Publisher("auv0_command", std_msgs.msg.Float32MultiArray, queue_size=1)
        diver_command_pub = rospy.Publisher("diver_command", std_msgs.msg.Float32MultiArray, queue_size=1)

        init_time = rospy.get_time()


        # diver_actors = []
        # for i in range(100):
        #     diver_actors.append(Actor([0.0, 0.0, -10.0], Rotation.from_euler('xyz', [0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0], dtype=np.float32), init_time))
        auv0_actor = Actor([5.0, 0.0, -10.0], Rotation.from_euler('xyz', [0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0], dtype=np.float32), init_time)
        diver_actor = Actor([0.0, 0.0, -10.0], Rotation.from_euler('xyz', [0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0], dtype=np.float32), rospy.get_time())

        rospy.Subscriber("auv0_IMUSensor", sensor_msgs.msg.Imu, imu_callback, auv0_actor)
        rospy.Subscriber("diver_IMUSensor", sensor_msgs.msg.Imu, imu_callback, diver_actor)
        #rospy.Subscriber("diver_IMUSensor", sensor_msgs.msg.Imu, imu_callback_test, diver_actors)

        #rospy.Subscriber("auv0_AcousticBeaconSensor", AcousticBeaconSensor, auv0_acoustic_callback)

        # diver_imu_sub = message_filters.Subscriber("diver_IMUSensor", sensor_msgs.msg.Imu)
        # auv0_imu_sub = message_filters.Subscriber("auv0_IMUSensor", sensor_msgs.msg.Imu)

        # ts = message_filters.TimeSynchronizer([diver_imu_sub, auv0_imu_sub], 10)
        # ts.registerCallback(imu_callback)
        start_time = rospy.get_time()
        while True:
            if 'p' in controller.pressed_keys:
                break
            now = rospy.get_time()
            dt = now - prev_time
            prev_time = now

            state = env.tick()

            diver_actor.orientation = Rotation.from_euler('xyz', state["diver"]['RotationSensor'] * (np.pi / 180.0))
            auv0_actor.orientation = Rotation.from_euler('xyz', state["auv0"]['RotationSensor'] * (np.pi / 180.0))

            ros_wrapper.publish_sensors(state)

            print("\n")

            print("auv0 pos error: ", auv0_actor.position - state["auv0"]['LocationSensor'])
            print("auv0 rot error: ", auv0_actor.orientation.as_euler('xyz') - ((state["auv0"]['RotationSensor'] / 180.0) * np.pi))
            print("diver pos error: ", diver_actor.position - state["diver"]['LocationSensor'])
            print("diver rot error: ", diver_actor.orientation.as_euler('xyz') - ((state["diver"]['RotationSensor'] / 180.0) * np.pi))
            # errors = []
            # for actor in diver_actors:
            #     errors.append(np.linalg.norm(actor.position - state["diver"]['LocationSensor']))
            #print("min error: ", min(errors), "index: ", errors.index(min(errors)))
            #print("max error: ", max(errors), "index: ", errors.index(max(errors)))
            # rotate goal pos
            goal_pos_diver = desired_pos(start_time, now, 50.0)

            # calculate goal pos in world frame
            goal_pos_world = np.array(diver_actor.position) + goal_pos_diver

            # calculate goal pos in auv frame
            goal_pos_auv = (goal_pos_world - np.array(auv0_actor.position))

            sum_error += goal_pos_auv * dt

            d_error = (goal_pos_auv - prev_error) / dt
            prev_error = goal_pos_auv

            auv_command = k_p * goal_pos_auv + k_d * d_error + k_i * sum_error
            #print(auv_command)
            #auv0_command_pub.publish(std_msgs.msg.Float32MultiArray(data = np.append([0.0, 0.0, -10.0],[0.0, 0.0, 0.0])))
            auv0_command_pub.publish(std_msgs.msg.Float32MultiArray(data = np.append(auv_command,[0.0, 0.0, 0.0])))

            #env.send_acoustic_message(0, 1, "MSG_RESPX", 'howdy!')

            diver_command = controller.parse_keys(20)
            diver_command_pub.publish(std_msgs.msg.Float32MultiArray(data = diver_command))
#
#
# THIS IS UNTESTED AND SHOULD BE USED AS A REFERENCE FOR CREATING YOUR OWN ROS PACKAGE ONLY
#
#

from holoocean.environments import HoloOceanEnvironment

import rospy
import numpy as np
from scipy.spatial.transform import Rotation

import std_msgs.msg
import sensor_msgs.msg
import geometry_msgs.msg
from holo_wrapper.msg import AcousticBeaconSensor

import HoloOceanUtils.automated_experiments.controller as controller
from HoloOceanUtils.factor_graph.actor import Actor
from HoloOceanUtils.ROS.holoocean_ros_wrapper import ROSWrapper

rospy.init_node("wrapper_node", anonymous=True)

config = {
    "package_name": "MRG",
    "name": "Hovering",
    "world": "OpenOcean",
    "main_agent": "B",
    "agents": [
            {
            "agent_name": "A",
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
            "control_scheme": 2,
            "location": [0, 5, -10]
            },
            {
            "agent_name": "B",
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
    "window_height": 720,
    "ticks_per_sec": 300,
    "frames_per_sec": True
}

def imu_callback(msg, actor):
    dt = 1.0 / float(config["ticks_per_sec"])

    lin_accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
    ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

    actor.integrate_imu_paper(lin_accel, ang_vel, dt)

if __name__ == "__main__":
    with HoloOceanEnvironment(scenario_cfg=config) as env:
        env.reset()

        # controller variables
        parameters = {
             "circling_period": 50.0,
             "circling_radius": 15.0
        }
        prev_error = [0.0, 0.0, 0.0]
        sum_error = [0.0, 0.0, 0.0]

        # this sets up the ROSWrapper's publishers and subscribers based on the holoocean environment
        ros_wrapper = ROSWrapper(env)

        # init actors
        auv0_actor = Actor(np.array([5.0, 0.0, -10.0]), Rotation.from_euler('xyz', [0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), (0.0, 0.0), "A")
        diver_actor = Actor(np.array([0.0, 0.0, -10.0]), Rotation.from_euler('xyz', [0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), (0.0, 0.0), "A")

        # init command publishers and IMU sensor subscribers
        auv0_command_pub = rospy.Publisher("A_command", std_msgs.msg.Float32MultiArray, queue_size=1)
        diver_command_pub = rospy.Publisher("B_command", std_msgs.msg.Float32MultiArray, queue_size=1)
        rospy.Subscriber("A_IMUSensor", sensor_msgs.msg.Imu, imu_callback, auv0_actor)
        rospy.Subscriber("B_IMUSensor", sensor_msgs.msg.Imu, imu_callback, diver_actor)

        start_time = rospy.get_time()
        counter = 0

        while True:
            dt = 1.0 / float(config["ticks_per_sec"])
            now = start_time + (counter * dt)

            # example acoustic message
            env.send_acoustic_message(0, 1, "MSG_RESPX", 'howdy!')
            state = env.tick()

            # this publishes the sensor messages we recieved from the tick function
            ros_wrapper.publish_sensors(state)

            # publish the auv command
            auv_command = controller.auv_command(parameters, state, prev_error, sum_error, dt, start_time, now)
            auv0_command_pub.publish(std_msgs.msg.Float32MultiArray(data = np.append(auv_command,[0.0, 0.0, 0.0])))

            # publish the diver command
            diver_command = controller.choose_random_commands(10, np.random)
            diver_command_pub.publish(std_msgs.msg.Float32MultiArray(data = diver_command))

            counter += 1
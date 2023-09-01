import rospy
import numpy as np

import std_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
from holo_wrapper.msg import AcousticBeaconSensor

class ROSWrapper:
    def __init__(self, env):
        """
        Init publishers for sensor messages
        """
        def command_actor(msg, name):
            env.act(name, np.array(msg.data))

        self.publishers = {}
        for agent_name in env.agents:
            agent = env.agents[agent_name]
            rospy.Subscriber(agent_name + "_command", std_msgs.msg.Float32MultiArray, command_actor, agent_name)

            for _, sensor in agent.sensors.items():
                sensor_name = agent_name + "_" + sensor.name

                if sensor.name == "LocationSensor":
                    self.publishers[sensor_name] = \
                        rospy.Publisher(sensor_name, geometry_msgs.msg.PointStamped, queue_size=1)

                elif sensor.name == "CollisionSensor":
                    self.publishers[sensor_name] = \
                        rospy.Publisher(sensor_name, std_msgs.msg.Bool, queue_size=1)

                elif sensor.name == "IMUSensor":
                    self.publishers[sensor_name] = \
                        rospy.Publisher(sensor_name, sensor_msgs.msg.Imu, queue_size=1)
                    self.publishers[sensor_name + "_bias"] = \
                        rospy.Publisher(sensor_name + "_bias", sensor_msgs.msg.Imu, queue_size=1)

                elif sensor.name == "AcousticBeaconSensor":
                    self.publishers[sensor_name] = \
                        rospy.Publisher(sensor_name, AcousticBeaconSensor, queue_size=1)

    def publish_sensors(self, state):
        """
        Publish sensor messages
        """
        for agent in state:
            if agent != 't':
                #print(state[agent])
                for sensor in state[agent]:
                    sensor_name = agent + "_" + sensor
                    if sensor == "LocationSensor":
                        msg = geometry_msgs.msg.PointStamped()

                        msg.header = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id=agent)
                        msg.point.x = state[agent][sensor][0]
                        msg.point.y = state[agent][sensor][1]
                        msg.point.z = state[agent][sensor][2]

                        self.publishers[sensor_name].publish(msg)

                    elif sensor == "CollisionSensor":
                        msg = std_msgs.msg.Bool()

                        msg.data = state[agent][sensor]

                        self.publishers[sensor_name].publish(msg)

                    elif sensor == "IMUSensor":
                        msg = sensor_msgs.msg.Imu()

                        msg.header = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id=agent)
                        #print(state[agent][sensor])
                        msg.linear_acceleration.x = state[agent][sensor][0][0]
                        msg.linear_acceleration.y = state[agent][sensor][0][1]
                        msg.linear_acceleration.z = state[agent][sensor][0][2]

                        msg.angular_velocity.x = state[agent][sensor][1][0]
                        msg.angular_velocity.y = state[agent][sensor][1][1]
                        msg.angular_velocity.z = state[agent][sensor][1][2]

                        self.publishers[sensor_name].publish(msg)

                        msg_bias = sensor_msgs.msg.Imu()

                        msg_bias.header = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id=agent)
                        msg_bias.linear_acceleration.x = state[agent][sensor][2][0]
                        msg_bias.linear_acceleration.y = state[agent][sensor][2][1]
                        msg_bias.linear_acceleration.z = state[agent][sensor][2][2]

                        msg_bias.angular_velocity.x = state[agent][sensor][3][0]
                        msg_bias.angular_velocity.y = state[agent][sensor][3][1]
                        msg_bias.angular_velocity.z = state[agent][sensor][3][2]

                        self.publishers[sensor_name + "_bias"].publish(msg_bias)

                    elif sensor == "AcousticBeaconSensor":
                        msg = AcousticBeaconSensor()

                        msg.header = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id=agent)
                        msg.message_type = state[agent][sensor][0]
                        msg.id = state[agent][sensor][1]
                        msg.payload = state[agent][sensor][2]
                        msg.azimuth = state[agent][sensor][3]
                        msg.elevation = state[agent][sensor][4]
                        msg.range = state[agent][sensor][5]
                        msg.depth = state[agent][sensor][6]

                        self.publishers[sensor_name].publish(msg)
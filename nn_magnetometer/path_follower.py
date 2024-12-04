import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
import numpy as np
import math
from ament_index_python.packages import get_package_share_directory
from rclpy.duration import Duration
from ament_index_python.packages import get_package_share_directory

def quaternion_to_euler(z, w):
    t0 = +2.0 * (w * z + 0.0 * 0.0)
    t1 = +1.0 - 2.0 * (0.0 * 0.0 + z * z)
    euler_z = math.atan2(t0, t1)

    return euler_z


class PathFollower(Node):

    def __init__(self):
        super().__init__("magnav_entmap")
        robot_namespace = self.declare_parameter("robot_namespace", "robot").value
        self.linear_velocity = self.declare_parameter("linear_velocity", 0.1).value
        self.k_stanley = self.declare_parameter("stanley_gain", 0.5).value
        flip = self.declare_parameter("flip_path", False).value

        self.mocap_pose = self.create_subscription(
            PoseStamped, f"/{robot_namespace}/enu/pose", self.mocap_callback, 1
        )
        self.estimated_pose = self.create_subscription(
            msg_type=PoseWithCovarianceStamped,
            topic=f"/turtle/estimated_pose",
            callback=self.estimated_callback,
            qos_profile=1,
        )
        self.cmd_vel = self.create_publisher(Twist, f"/{robot_namespace}/cmd_vel", 1)
        share_directory = get_package_share_directory("nn_magnetometer")
        path_file = share_directory+self.declare_parameter("path_file", "skinny_figure8_path.csv").value
        self.path = np.genfromtxt(
            path_file,
            delimiter=",",
        )
        self.path = np.flip(self.path, 0) if flip else self.path
        self.loop = self.declare_parameter("do_loop", False).value
        continuous = self.declare_parameter("continuous", False).value
        flipped_path = np.flip(self.path, 0)
        flipped_path[:, 1] = np.flip(flipped_path[:, 1])
        self.path = np.vstack((self.path, flipped_path)) if not continuous else self.path
        self.control = self.create_timer(1 / 50, self.control_callback)
        self.mocap_pose = np.zeros((3, 1))
        self.estimated_pose = np.zeros((3, 1))
        self.i = 0
        self.get_clock().sleep_for(Duration(seconds=2.0))

    def mocap_callback(self, msg):
        self.mocap_pose = np.array(
            [
                [
                    msg.pose.position.x,
                    msg.pose.position.y,
                    quaternion_to_euler(msg.pose.orientation.z, msg.pose.orientation.w),
                ]
            ]
        ).T

    def estimated_callback(self, msg):
        self.estimated_pose = np.array(
            [
                [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    quaternion_to_euler(
                        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
                    ),
                ]
            ]
        ).T

    def control_callback(self):
        # self.get_logger().info(f"Estimated Pose: {self.estimated_pose.flatten()}")
        # closest_point, cross_track = self.closest_point_on_path(
        #     self.path, self.estimated_pose.flatten()
        # )
        if not self.use_estimate:
            self.estimated_pose = self.mocap_pose
        closest_point = self.path[self.i, :]
        self.get_logger().info(f"Closest Point is  :{closest_point}")
        cross_track = np.sqrt(
            (closest_point[0] - self.estimated_pose[0]) ** 2
            + (closest_point[1] - self.estimated_pose[1]) ** 2
        )
        if cross_track < 0.25:
            self.i += 1
        if self.loop and self.i >= self.path.shape[0] - 1:
            self.i = 0
        angle_to_point = np.arctan2(
            closest_point[1] - self.estimated_pose[1],
            closest_point[0] - self.estimated_pose[0],
        )
        angle_error = angle_to_point - self.estimated_pose[2]

        angle_error = self.angle_wrap(angle_error)
        omega = angle_error + math.atan2(
            self.k_stanley * cross_track, self.linear_velocity
        )
        msg_out = Twist()
        msg_out.linear.x = (
            self.linear_velocity if self.i <= self.path.shape[0] - 1 else 0.0
        )
        msg_out.angular.z = float(omega) if self.i <= self.path.shape[0] - 1 else 0.5
        self.cmd_vel.publish(msg_out)

    @staticmethod
    def closest_point_on_path(path, current_pose):
        current_pose = np.array(current_pose)  # Ensure it's a 1D array
        # Calculate distances from current_pose to each point in the path
        distances = np.sqrt(
            (path[:, 0] - current_pose[0]) ** 2 + (path[:, 1] - current_pose[1]) ** 2
        )
        # Find the index of the minimum distance
        closest_index = np.argmin(distances)

        # Return the closest point and the distance to it
        return path[closest_index], distances[closest_index]

    @staticmethod
    def angle_wrap(angle):
        """
        Wrap the angle to be within the range [-pi, pi].
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


def main(args=None):
    rclpy.init()
    node = PathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

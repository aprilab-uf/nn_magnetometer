from typing import Optional
import json
import rclpy
from rclpy.node import Node
import numpy as np
import os
import time
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sklearn.preprocessing import StandardScaler
from sensor_msgs.msg import Imu, BatteryState, MagneticField
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Make sure to import SimpleNN from the correct location
from .magnav_nn import SimpleNN  # Import SimpleNN class here


def quaternion_to_euler(z, w):
    t0 = +2.0 * (w * z + 0.0 * 0.0)
    t1 = +1.0 - 2.0 * (0.0 * 0.0 + z * z)
    euler_z = math.atan2(t0, t1)

    return euler_z


class Frame:
    timestamp: float
    true_magnetic_magnitude: float
    nn_magnetic_magnitude: float
    map_magnetic_magnitude: float
    pose_data: tuple[float, float, float]
    # vector_field: tuple[float, float, float]
    odom: tuple[float, float, float, float, float, float]
    imu_data: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    battery_data: tuple[float, float]
    lidar: int


ENTRIES = (
    "timestamp",
    "true_magnetic_magnitude",
    "nn_magnetic_magnitude",
    "map_magnetic_magnitude",
    "pose_data",
    # "vector_field",
    "odom",
    "imu_data",
    "battery_data",
    "lidar"
)


class NNMagnetometer(Node):
    current_frame: Frame

    def __init__(self):
        super().__init__("raph_subscriber")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.robot_namespace = self.declare_parameter("robot_namespace","robot").value
        # Initialize the model (make sure SimpleNN is imported above)
        self.model = SimpleNN(20, 1)
        # Load the model weights (ensure the model class is defined before loading the weights)
        model_path = self.declare_parameter("model_path","").value
        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.scaler_X = checkpoint["scaler_X"]
        self.scaler_y = checkpoint["scaler_y"]
        self.model.to(self.device)
        self.model.eval()

        # Create a timestamped folder based on the slowest subscriber
        data_type = self.declare_parameter("data_type","").value
        self.folder_name = self.create_timestamped_folder(data_type=data_type)
        self.filename = os.path.join(self.folder_name, f"{self.robot_namespace}_test_data.json")

        # Initialize a list to store sensor data
        self.current_frame = Frame()

        # Subscribe to various topics
        self.pose_sub = self.create_subscription(
            PoseStamped, f"/{self.robot_namespace}/enu/pose", self.pose_callback, 10
        )
        self.cmd_vel_sub = self.create_subscription(
            Odometry, f"/{self.robot_namespace}/odom", self.cmd_vel_callback, 10
        )
        self.imu_sub = self.create_subscription(Imu, f"/{self.robot_namespace}/imu", self.imu_callback, 10)
        self.mag_intensity_pub = self.create_publisher(
            MagneticField, f"/{self.robot_namespace}/qtfm/magnitude", 10
        )
        self.battery_timer = self.create_timer(2,self.battery_state_callback)
        self.mag_timer = self.create_timer(1/135,self.magnitude_callback)
        self.i = 0
        self.battery_percentage = 0.67
        self.battery_decrement = 0.01
        lidar = self.declare_parameter("lidar",0).value
        self.current_frame.lidar = lidar

        # Initialize interpolation map
        self.interpolate_map()
        self.save = True

    def create_timestamped_folder(self,data_type=""):
        timestamp = datetime.now().strftime("%d%m%yT%H%M%S")
        folder_name = f"{timestamp}_test_data_{self.robot_namespace}_{data_type}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        self.get_logger().info(f"Created folder: {folder_name}")
        return folder_name

    def save_data_to_file(self):
        if self.current_frame.timestamp is None:
            self.get_logger().warn("No data received, skipping save.")
            return

        np_data = {entry: getattr(self.current_frame, entry) for entry in ENTRIES}
        np_filename = os.path.join(self.folder_name, f"{self.robot_namespace}_test_data.json")
        with open(np_filename, "a") as f:
            json.dump(np_data, f)
            f.write(",\n")

    def cmd_vel_callback(self, msg):
        timestamp = time.time()
        self.current_frame.odom = (msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.linear.z,msg.twist.twist.angular.x,msg.twist.twist.angular.y,msg.twist.twist.angular.z)
        self.current_frame.timestamp = timestamp
        self.save_data()

    def magnitude_callback(self):
        timestamp = time.time()
        self.current_frame.true_magnetic_magnitude =0.0 
        self.current_frame.timestamp = timestamp
        self.current_frame.nn_magnetic_magnitude = 0.0
        self.current_frame.map_magnetic_magnitude = 0.0
        if all(hasattr(self.current_frame, attr) for attr in ENTRIES):
            pose = list(self.current_frame.pose_data)
            pose[2] = self.wrap_angle(pose[2] * (180 / np.pi))
            b_e = self.interp_map(pose[::-1])[0]
            self.current_frame.map_magnetic_magnitude = b_e
            with torch.no_grad():
                flattened_entry = [
                    pose[2],
                    self.current_frame.odom[0],
                    self.current_frame.odom[1],
                    self.current_frame.odom[2],
                    self.current_frame.odom[3],
                    self.current_frame.odom[4],
                    self.current_frame.odom[5],
                    self.current_frame.imu_data[0][0],
                    self.current_frame.imu_data[0][1],
                    self.current_frame.imu_data[0][2],
                    self.current_frame.imu_data[0][3],
                    self.current_frame.imu_data[1][0],
                    self.current_frame.imu_data[1][1],
                    self.current_frame.imu_data[1][2],
                    self.current_frame.imu_data[2][0],
                    self.current_frame.imu_data[2][1],
                    self.current_frame.imu_data[2][2],
                    self.current_frame.battery_data[0],
                    self.current_frame.battery_data[2],
                    self.current_frame.lidar,
                ]
                X_scaled = self.scaler_X.transform(
                    [flattened_entry]
                )  # Fit and transform

                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                predictions = self.model(X_tensor.to(self.device))

                predictions_rescaled = self.scaler_y.inverse_transform(
                    predictions.cpu().numpy()
                )

            nn_mag_field = b_e + predictions_rescaled[0][0]
            self.current_frame.nn_magnetic_magnitude = nn_mag_field
            self.current_frame.map_magnetic_magnitude = b_e
        self.save_data()

    def pose_callback(self, msg):
        timestamp = time.time()
        self.current_frame.pose_data = (
            msg.pose.position.x,
            msg.pose.position.y,
            quaternion_to_euler(msg.pose.orientation.z, msg.pose.orientation.w),
        )
        self.current_frame.timestamp = timestamp
        self.save_data()


    def imu_callback(self, msg):
        timestamp = time.time()
        self.current_frame.imu_data = (
            (
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ),
            (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z),
            (
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ),
        )
        self.current_frame.timestamp = timestamp
        self.save_data()

    def battery_state_callback(self):
        timestamp = time.time()
        self.battery_percentage -= self.battery_decrement
        self.current_frame.battery_data = (np.random.normal(14.55,0.02), 1.0, self.battery_percentage)
        self.current_frame.timestamp = timestamp
        self.save_data()

    def save_data(self):
        if all(hasattr(self.current_frame, attr) for attr in ENTRIES) and self.save:
            self.save_data_to_file()

    def wrap_angle(self, angle):
        wrapped_angle = angle % 360
        if wrapped_angle < 0:
            wrapped_angle += 360
        return wrapped_angle

    def interpolate_map(self):
        # Load map data and create interpolator
        ninety_map = np.loadtxt(
            "/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/map/90_map.csv",
            delimiter=",",
        )
        zero_map = np.loadtxt(
            "/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/map/0_map.csv",
            delimiter=",",
        )
        oneeighty_map = np.loadtxt(
            "/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/map/180_map.csv",
            delimiter=",",
        )
        twoseventy_map = np.loadtxt(
            "/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/map/270_map.csv",
            delimiter=",",
        )
        self.map = np.vstack(
            (zero_map, ninety_map, oneeighty_map, twoseventy_map, zero_map)
        )

        # print(map)
        x = np.unique(self.map[:, 0])
        y = np.unique(self.map[:, 1])
        theta = np.array([0, 90, 180, 270, 360])
        values = self.map[:, 3].reshape((len(theta), len(y), len(x)))
        self.interp_map = RegularGridInterpolator(points=(theta, y, x), values=values)


def main(args=None):
    rclpy.init(args=args)
    node = NNMagnetometer()
    with open(node.filename, "w") as f:
        f.write("[\n")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        with open(node.filename, "r+") as f:
            pos = f.seek(0, 2)
            f.seek(pos - 2)
            f.write("]")
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

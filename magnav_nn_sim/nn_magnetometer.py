import json
import rclpy
from rclpy.node import Node
import numpy as np
import os
import time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, BatteryState, MagneticField
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime
import math
import torch
from .SimpleNN import SimpleNN  


def quaternion_to_euler(z, w):
    t0 = +2.0 * (w * z + 0.0 * 0.0)
    t1 = +1.0 - 2.0 * (0.0 * 0.0 + z * z)
    euler_z = math.atan2(t0, t1)

    return euler_z


class Frame:
    """ This class is used to store the data from the various sensors. It is also used to predict the magnetic field using the neural network model. 
    Parameters:
                timestamp: float - The timestamp of the data
                nn_magnetic_magnitude: float - The predicted magnetic field magnitude
                map_magnetic_magnitude: float - The magnetic field magnitude from the map
                pose_data: tuple[float, float, float] - The x, y, and yaw of the robot
                odom: tuple[float, float, float, float, float, float] - The linear x, y, z, and angular x, y, z velocities
                imu_data: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] - The orientation, angular velocity, and linear acceleration
                battery_data: tuple[float, float] - The voltage and percentage of the battery
                lidar: int - The lidar status"""
    timestamp: float
    nn_magnetic_magnitude: float
    map_magnetic_magnitude: float
    pose_data: tuple[float, float, float]
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
    "nn_magnetic_magnitude",
    "map_magnetic_magnitude",
    "pose_data",
    "odom",
    "imu_data",
    "battery_data",
    "lidar"
)


class NNMagnetometer(Node):
    """This class is used to subscribe to various topics and predict the magnetic field using a neural network model. The data is saved to a JSON file if desired."""
    current_frame: Frame

    def __init__(self):
        super().__init__("raph_subscriber")
        # Get the parameters
        self.robot_namespace = self.declare_parameter("robot_namespace","robot").value # The robot namespace
        model_path = self.declare_parameter("model_name","nn_magnetometer_model_v0.0.1.pth").value # The path to the model
        self.save_json = self.declare_parameter("save_json",False).value # Whether to save the data to a JSON file
        self.sim = self.declare_parameter("is_sim",True).value # Whether the robot is in simulation
        data_type = self.declare_parameter("data_type","").value

        # Initialize the model and the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleNN(20, 1)
        # Load the model weights and scalers

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
        if self.save_json:
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
        self.mag_intensity_pub = self.create_publisher(
            MagneticField, f"/{self.robot_namespace}/qtfm/magnitude", 10
        )
        self.imu_sub = self.create_subscription(Imu, f"/{self.robot_namespace}/imu", self.imu_callback, 10)

        if not self.sim:
            self.battery_state_sub = self.create_subscription(
                BatteryState, f"/{self.robot_namespace}/battery_state", self.battery_state_callback, 10)
        else:
            self.battery_timer = self.create_timer(2,self.battery_timer_callback)

        self.mag_timer = self.create_timer(1/135,self.magnitude_callback)
        self.i = 0

        # Initialize the battery percentage and decrement for simulation
        self.battery_percentage = 0.67
        self.battery_decrement = 0.01

        # Set the lidar status
        lidar = self.declare_parameter("lidar",0).value
        self.current_frame.lidar = lidar

        # Initialize interpolation map
        self.interpolate_map()

    def create_timestamped_folder(self,data_type=""):
        """
        Create a timestamped folder to store the data

        Parameters:
                    data_type: str - The type of data being stored
        Returns:
                    folder_name: str - The name of the folder created
        """

        timestamp = datetime.now().strftime("%d%m%yT%H%M%S")
        folder_name = f"{timestamp}_test_data_{self.robot_namespace}_{data_type}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        self.get_logger().info(f"Created folder: {folder_name}")
        return folder_name

    def save_data_to_file(self):
        """Save the data to a JSON file"""

        if self.current_frame.timestamp is None or self.save_json is False:
            self.get_logger().warn("No data received, skipping save or JSON saving is disabled")
            return

        np_data = {entry: getattr(self.current_frame, entry) for entry in ENTRIES}
        np_filename = os.path.join(self.folder_name, f"{self.robot_namespace}_test_data.json")
        with open(np_filename, "a") as f:
            json.dump(np_data, f)
            f.write(",\n")
    
    def save_data(self):
        """Save the data to a JSON file if all the data is present"""

        if all(hasattr(self.current_frame, attr) for attr in ENTRIES) and self.save_json:
            self.save_data_to_file()

    def cmd_vel_callback(self, msg):
        """
        Callback function for the odometry data

        Parameters:
                    msg: Odometry - The odometry message
        """
        timestamp = time.time()
        self.current_frame.odom = (msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.linear.z,msg.twist.twist.angular.x,msg.twist.twist.angular.y,msg.twist.twist.angular.z)
        self.current_frame.timestamp = timestamp
        self.save_data()

    def magnitude_callback(self):
        """Callback function for the magnitude prediction"""

        timestamp = time.time()
        self.current_frame.timestamp = timestamp
        # Setting the magnetic field magnitude to 0.0 to ensure the data frame is complete
        self.current_frame.nn_magnetic_magnitude = 0.0
        self.current_frame.map_magnetic_magnitude = 0.0
        if all(hasattr(self.current_frame, attr) for attr in ENTRIES):
            flattened_entry = self.get_flattened_array()
            with torch.no_grad():

                X_scaled = self.scaler_X.transform(
                    [flattened_entry]
                )  # Fit and transform

                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                predictions = self.model(X_tensor.to(self.device))

                predictions_rescaled = self.scaler_y.inverse_transform(
                    predictions.cpu().numpy()
                )

            nn_mag_field = self.current_frame.map_magnetic_magnitude + predictions_rescaled[0][0]
            self.current_frame.nn_magnetic_magnitude = nn_mag_field
            msg = MagneticField()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.magnetic_field.x = nn_mag_field
            self.mag_intensity_pub.publish(msg)
        self.save_data()

    def pose_callback(self, msg):
        """
        Callback function for the pose data
        
        Parameters:
                    msg: PoseStamped - The pose message
        """
        timestamp = time.time()
        self.current_frame.pose_data = (
            msg.pose.position.x,
            msg.pose.position.y,
            quaternion_to_euler(msg.pose.orientation.z, msg.pose.orientation.w),
        )
        self.current_frame.timestamp = timestamp
        self.save_data()


    def imu_callback(self, msg):
        """
        Callback function for the IMU data

        Parameters:
                    msg: Imu - The IMU message
        """

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

    def battery_state_callback(self,msg):
        """
        Callback function for the battery state data

        Parameters:
                    msg: BatteryState - The battery state message
        """

        timestamp = time.time()
        self.current_frame.battery_data = (msg.voltage, msg.percentage)
        self.current_frame.timestamp = timestamp
        self.save_data()
        
    def battery_timer_callback(self):
        """
        Callback function for the battery timer
        """
        timestamp = time.time()
        self.battery_percentage -= self.battery_decrement
        self.current_frame.battery_data = (np.random.normal(14.55,0.02), self.battery_percentage)
        self.current_frame.timestamp = timestamp
        self.save_data()
   
    def get_flattened_array():
        """Get the flattened array of the data frame"""

        pose = list(self.current_frame.pose_data)
        pose[2] = self.wrap_angle(pose[2] * (180 / np.pi))
        b_e = self.interp_map(pose[::-1])[0]
        self.current_frame.map_magnetic_magnitude = b_e
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
            self.current_frame.battery_data[1],
            self.current_frame.lidar,
        ]
        return flattened_entry

    def interpolate_map(self):
        """Interpolate the magnetic field map"""
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

        x = np.unique(self.map[:, 0])
        y = np.unique(self.map[:, 1])
        theta = np.array([0, 90, 180, 270, 360])
        values = self.map[:, 3].reshape((len(theta), len(y), len(x)))
        self.interp_map = RegularGridInterpolator(points=(theta, y, x), values=values)

    @staticmethod
    def wrap_angle(angle):
        """Wrap the angle to be between 0 and 360 degrees
        
        Parameters:
                    angle: float - The angle to be wrapped
        Returns:
                    wrapped_angle: float - The wrapped angle
        """

        wrapped_angle = angle % 360
        if wrapped_angle < 0:
            wrapped_angle += 360
        return wrapped_angle


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

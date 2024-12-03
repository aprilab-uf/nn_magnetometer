import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from nflows.transforms import AffineCouplingTransform, CompositeTransform, MaskedAffineAutoregressiveTransform
from nflows.distributions import StandardNormal
from nflows.flows import Flow
import json
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


class INN(nn.Module):
    def __init__(self, input_dim, num_layers=4, hidden_features=128):
        super(INN, self).__init__()
        # Build a sequence of affine coupling layers for the INN
        transforms = []
        for _ in range(num_layers):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=input_dim,
                    hidden_features=hidden_features,
                    num_blocks=2,
                )
            )
        self.flow = Flow(transform=CompositeTransform(transforms), distribution=StandardNormal([input_dim]))

    def forward(self, x):
        # Forward transform: map input to latent space
        z, log_jac_det = self.flow._transform.forward(x)
        return z, log_jac_det

    def inverse(self, z):
        # Inverse transform: map latent space back to input
        x, log_jac_det = self.flow._transform.inverse(z)
        return x, log_jac_det


class TrainData:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.interpolate_map()

    def interpolate_map(self):
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

    def flatten_data_to_array(self, data_path):
        with open(
            data_path,
            "r",
        ) as f:
            data = json.load(f)
        flattened_data = []
        for entry in data:
            # Flatten the dictionaries
            odom = entry["odom"]
            imu_data = entry["imu_data"]
            battery_data = entry["battery_data"]
            vector_field = (
                entry["vector_field"]
                if entry["vector_field"]
                else {"x": 0.0, "y": 0.0, "z": 0.0}
            )
            pose = entry["pose_data"]
            pose[2] = self.wrap_angle(pose[2] * (180 / np.pi))
            lidar = entry["lidar"]

            b_e = self.interp_map(pose[::-1])
            # Create a flattened list of values
            flattened_entry = [
                entry["true_magnetic_magnitude"] - b_e[0],  # Magnetic magnitude
                # pose[0], # Position X
                # pose[1], # Position Y
                pose[2],  # Orientation Yaw
                odom[0],  # odom Linear velocity x
                odom[1],  # odom Linear velocity y
                odom[2],  # odom Linear velocity z
                odom[3],  # odom Angular velocity x
                odom[4],  # odom Angular velocity y
                odom[5],  # odom Angular velocity z
                imu_data[0][0],  # IMU compass heading
                imu_data[0][1],  # IMU compass heading
                imu_data[0][2],  # IMU compass heading
                imu_data[0][3],  # IMU compass heading
                imu_data[1][0],  # IMU angular velocity X
                imu_data[1][1],  # IMU angular velocity Y
                imu_data[1][2],  # IMU angular velocity Z
                imu_data[2][0],  # IMU linear acceleration X
                imu_data[2][1],  # IMU linear acceleration Y
                imu_data[2][2],  # IMU linear acceleration Z
                battery_data[0],  # Battery voltage
                battery_data[2],  # Battery percentage
                lidar,  # LiDar On/Off
            ]
            flattened_data.append(flattened_entry)

        # Convert the flattened data list to a NumPy array
        flattened_data_np = np.array(flattened_data)
        return flattened_data_np
    def train(self, data, output_name="inn_model"):
        X = data[:, 1:]  # Input features
        y = data[:, 0]  # Target output

        # Normalize the features and target
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

        # Prepare dataset
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Initialize the INN
        input_dim = X.shape[1] + 1  # Input + output
        model = INN(input_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Training loop
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for X_batch, y_batch in dataloader:
                # Combine inputs and targets into a single tensor
                inputs = torch.cat((X_batch, y_batch), dim=1)

                # Forward pass
                z, log_jac_det = model(inputs)
                prior_loss = 0.5 * torch.sum(z**2, dim=1).mean()  # Standard normal prior
                jacobian_loss = -log_jac_det.mean()
                loss = prior_loss + jacobian_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

        # Save the trained model and scalers
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
            },
            f"{output_name}.pth",
        )
        print("Model saved.")

    def eval(self, data, model_path):
        checkpoint = torch.load(model_path)
        model = INN(data.shape[1]).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        scaler_X = checkpoint["scaler_X"]
        scaler_y = checkpoint["scaler_y"]

        model.eval()
        X = data[:,1:]  # Use only the input features
        X_scaled = scaler_X.transform(X)
        y = data[:, 0]


        inputs = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            z, _ = model(inputs)
            predictions, _ = model.inverse(z)
            predictions = predictions[:, -1].cpu().numpy()  # Extract the predicted target
            predictions_rescaled = scaler_y.inverse_transform(predictions.reshape(-1, 1))

            steps = np.arange(0, len(predictions_rescaled[:, 0]), 1)
            plt.figure()
            plt.plot(steps, predictions_rescaled[:, 0], "r--", label="Predictions")
            plt.plot(steps,y,"b",label = "Truth")
            plt.legend()
            plt.show()
    @staticmethod
    def wrap_angle(angle):
        # Wrap the angle between 0 and 360 degrees
        wrapped_angle = angle % 360  # Ensure the angle is between 0 and 360
        if wrapped_angle < 0:
            wrapped_angle += 360  # Handle negative angles, wrap into positive range
        return wrapped_angle
    
def main():
    import glob

    # Determine the device to use for training (CUDA or MPS or CPU)
    training = TrainData()

    data1 = training.flatten_data_to_array(
        data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V3_training/201124T154504_training_data_leo_figure8_lidar_off/leo_data.json"
    )
    data2 = training.flatten_data_to_array(
        data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V3_training/201124T154745_training_data_leo_figure8flipped_lidar_off/leo_data.json"
    )
    data3 = training.flatten_data_to_array(
        data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V3_training/201124T155024_training_data_leo_sinwave_lidar_off/leo_data.json"
    )
    data4 = training.flatten_data_to_array(
        data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V3_training/201124T155301_training_data_leo_sinwaveflipped_lidar_off/leo_data.json"
    )
    data5 = training.flatten_data_to_array(
        data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V3_training/201124T155829_training_data_leo_sinwaveflipped_lidar_on/leo_data.json"
    )
    data6 = training.flatten_data_to_array(
        data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V4.1_training/241124T094809_training_data_leo_manual_control_lidar_off/leo_data.json"
    )
    data7 = training.flatten_data_to_array(
        data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V3_training/201124T160245_training_data_leo_figure8_lidar_on/leo_data.json"
    )
    data8 = training.flatten_data_to_array(
        data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V3_training/201124T160446_training_data_leo_figure8flipped_lidar_on/leo_data.json"
    )
    data9 = training.flatten_data_to_array(
        data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V3_training/211124T124240_training_data_leo_lawnmower5_lidar_off/leo_data.json"
    )
    data10 = training.flatten_data_to_array(
        data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V3_training/211124T124840_training_data_leo_lawnmower5_lidar_on/leo_data.json"
    )
    data11 = training.flatten_data_to_array(
        data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V4.2_training/201124T165408_training_data_leo_highfreqSin_lidar_off/leo_data.json"
    )
    data12 = training.flatten_data_to_array(
        data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V4.2_training/201124T162736_training_data_leo_highfreqSin_lidar_on/leo_data.json"
    )
    data13 = training.flatten_data_to_array(
        data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V3_training/221124T135723_training_data_leo_lawnmower5_continous_lidar_off/leo_data.json"
    )
    # data14 = training.flatten_data_to_array(data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V3_Training/221124T142204_training_data_leo_lawnmowerSplitY_continous_lidar_off/leo_data.json")
    data15 = training.flatten_data_to_array(
        data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V4.1_training/241124T110327_training_data_leo_manual_control_lidar_off/leo_data.json"
    )
    data16 = training.flatten_data_to_array(
        data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V4.1_training/241124T113221_training_data_leo_manual_control_lidar_on/leo_data.json"
    )
    data17 = training.flatten_data_to_array(
        data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V4.2_training/211124T162931_training_data_leo_simplePath_lidar_off/leo_data.json"
    )
    data18 = training.flatten_data_to_array(
        data_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V4.1_training/251124T092252_training_data_leo_manual_control_lidar_on/leo_data.json"
    )

    data = np.vstack(
        (
            data1,
            data2,
            data3,
            data4,
            data5,
            data6,
            data7,
            data8,
            data9,
            data10,
            data11,
            data12,
            data13,
            data15,
            data16,
            data17,
            data18,
        )
    )
    print(data.shape)

    # training.train(data, output_name="mag_inn_model_v0.0.1")
    training.eval(data,model_path = "/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/models/mag_inn_model_v0.0.1.pth")


if __name__ == "__main__":
    main()

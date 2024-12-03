import torch
from scipy.stats import spearmanr  # Import Spearman correlation
import json
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import plotly.express as px
from torch.utils.tensorboard import SummaryWriter


class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(512, 128)
        self.output = nn.Linear(128, output_dim)
        self.scale = np.array([])
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # self.batch_norm1 = nn.BatchNorm1d(512)
        # self.batch_norm2 = nn.BatchNorm1d(512)
        # self.batch_norm3 = nn.BatchNorm1d(64)
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = F.relu(self.layer3(x))
        x = self.dropout(x)
        x = F.relu(self.layer4(x))
        x = self.output(x)
        return x

    # def forward(self, x):
    #     x = torch.relu(self.layer1(x))
    #     x = torch.relu(self.layer2(x))
    #     x = torch.relu(self.layer3(x))
    #     x = self.layer4(x)
    #     return x


class TrainData:
    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
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

    def train(self, data, output=False, output_name=""):
        output_path = f"data/models/{output_name}.pth"
        timestamp = datetime.now().strftime("%d%m%yT%H%M%S")
        writer = SummaryWriter(f"runs/{timestamp}_{output_name}")
        X = data[:, 1:]  # All columns except magnetic_magnitude
        y = data[:, 0]  # Magnetic magnitude is the target

        # Normalize the features and target using StandardScaler
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(
            y.reshape(-1, 1)
        )  # Reshape y to 2D for scaling
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).view(
            -1, 1
        )  # Reshape y to (n, 1)
        # Parameters for noise
        noise_mean = 0  # Mean of the noise
        x_noise_std = 0.01  # Standard deviation of the noise
        y_noise_std = 0.01  # Standard deviation for noise in the target

        x_noise = torch.normal(
            mean=noise_mean, std=x_noise_std, size=X_tensor.size()
        )  # Gaussian noise
        y_noise = torch.normal(
            mean=0, std=y_noise_std, size=y_tensor.size()
        )  # Gaussian noise
        y_noisy = y_tensor + y_noise  # Add the noise to the target
        X_noisy = X_tensor + x_noise
        # Create a DataLoader for batching
        batch_size = 32
        # dataset = TensorDataset(X_tensor, y_tensor)
        dataset = TensorDataset(X_noisy, y_noisy)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize the model, loss function, and optimizer
        input_dim = X.shape[1]  # Number of input features
        output_dim = 1  # For regression, output is a single value (magnetic_magnitude)

        model = SimpleNN(input_dim, output_dim).to(self.device)

        # Loss function (Mean Squared Error for regression)
        # criterion = nn.MSELoss()
        # Loss Function (Mean Average Error for regression)
        criterion = nn.MSELoss()

        # Optimizer (Adam optimizer)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # Training loop
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0

            # Iterate over the data in batches
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Compute the loss
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate the loss
                running_loss += np.sqrt(loss.detach().cpu().numpy())

            # Print the loss after each epoch
            loss_nt = scaler_y.inverse_transform(
                (running_loss / len(train_loader)).reshape((1, -1))
            ).item()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_nt:.4f}")
            writer.add_scalar("Loss/Train", loss_nt, epoch)
        writer.close()
        if output:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler_X": scaler_X,
                    "scaler_y": scaler_y,
                },
                output_path,
            )

        # Evaluate the model

    def eval(self, data, NNSize, model_path="", save_data=False):
        last_part = model_path.split("/")[-1]
        timestamp = datetime.now().strftime("%d%m%yT%H%M%S")
        # Load the model, scalers
        if not model_path:
            raise Exception("Model Path Cannot be empty")
        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )
        model = SimpleNN(NNSize, 1)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.scaler_X = checkpoint["scaler_X"]
        self.scaler_y = checkpoint["scaler_y"]

        model.to(self.device)
        model.eval()

        X = data[:, 1:]  # All columns except magnetic_magnitude
        y = data[:, 0]  # Magnetic magnitude is the target

        # Normalize the features using the saved scaler
        X_scaled = self.scaler_X.transform(X)
        y_scaled = self.scaler_y.transform(
            y.reshape(-1, 1)
        )  # Reshape y to 2D for scaling

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).view(
            -1, 1
        )  # Reshape y to (n, 1)

        with torch.no_grad():  # Disable gradient computation
            predictions = model(X_tensor.to(self.device))
            # Rescale the predictions and true values back to the original scale
            predictions_rescaled = self.scaler_y.inverse_transform(
                predictions.cpu().numpy()
            )
            true_values_rescaled = self.scaler_y.inverse_transform(
                y_tensor.cpu().numpy()
            )
            naive_predictions = 366.6620 * np.ones_like(predictions_rescaled)
            mae = np.mean(np.abs(predictions_rescaled - true_values_rescaled))
            naive_mae = np.mean(np.abs(naive_predictions - true_values_rescaled))
            print(f"Neural Network Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Naive Mean MAE: {naive_mae}")
            steps = np.arange(0, len(predictions_rescaled[:, 0]), 1)
            plt.figure()
            plt.plot(steps, predictions_rescaled[:, 0], "r--", label="Predictions")
            plt.plot(steps, true_values_rescaled[:, 0], "b--", label="Truth")
            plt.legend()
            plt.savefig("figs/SVG/predictions.svg")
            plt.savefig("figs/PDF/predictions.pdf")
            # plt.show()
            plt.figure()
            plt.plot(steps, true_values_rescaled[:, 0] - predictions_rescaled[:, 0])
            plt.savefig("figs/SVG/error.svg")
            plt.savefig("figs/PDF/error.pdf")
            # plt.show()
            if save_data:
                np.save(
                    f"{timestamp}_{last_part}.predicitons_data.npy",
                    np.array([steps, true_values_rescaled[:,0], predictions_rescaled[:,0]]),
                )

            # print("Predictions vs True Values (scaled back to original):")
            # for pred, true in zip(
            #     predictions_rescaled[:10], true_values_rescaled[:10]
            # ):  # Show first 10 predictions and true values
            #     print(f"Pred: {pred[0]:.4f}, True: {true[0]:.4f}")

    def plot_correlation(self, data):
        # Scale the data using StandardScaler
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        magnetic_magnitudes = data[:, 0]  # All rows, first column (magnetic magnitude)

        # Extract the other fields (all columns except the first)
        other_entries = data[:, 1:]  # All rows, columns from 2 to 20

        # Compute Spearman correlation between magnetic magnitude and each of the other fields
        correlations = []
        for i in range(other_entries.shape[1]):
            corr, _ = spearmanr(magnetic_magnitudes, other_entries[:, i])
            correlations.append(corr)

        labels = [
            "Pose X",
            "Pose Y",
            "Pose Theta",
            "Odom Linear Velocity x",  # odom Linear velocity x
            "Odom Linear Velocity y",  # odom Linear velocity y
            "Odom Linear Velocity z",  # odom Linear velocity z
            "Odom Angular Velocity x",  # odom Angular velocity x
            "Odom Angular Velocity y",  # odom Angular velocity y
            "Odom Angular Velocity z",  # odom Angular velocity z
            "IMU Compass Heading 1",  # IMU compass heading (1)
            "IMU Compass Heading 2",  # IMU compass heading (2)
            "IMU Compass Heading 3",  # IMU compass heading (3)
            "IMU Compass Heading 4",  # IMU compass heading (4)
            "IMU Angular Velocity X",  # IMU angular velocity X
            "IMU Angular Velocity Y",  # IMU angular velocity Y
            "IMU Angular Velocity Z",  # IMU angular velocity Z
            "IMU Linear Acceleration X",  # IMU linear acceleration X
            "IMU Linear Acceleration Y",  # IMU linear acceleration Y
            "IMU Linear Acceleration Z",  # IMU linear acceleration Z
            "Battery Voltage",  # Battery voltage
            "Battery Percentage",  # Battery percentage
            "LiDar On/Off",  # LiDar On/Off
        ]

        # Plot the correlations using a bar plot
        plt.figure(figsize=(18, 9))
        plt.bar(range(1, len(correlations) + 1), correlations, color="blue")
        plt.xlabel("Data Entries",fontsize = 18)
        plt.ylabel("Spearman Correlation",fontsize = 18)
        plt.xticks(
            range(1, len(correlations) + 1), labels, rotation=90
        ,fontsize = 18)  # Label the x-axis with meaningful names
        plt.yticks(fontsize = 18)
        plt.grid(True)
        plt.tight_layout()  # Ensure the labels fit without overlap
        plt.show()
        plt.savefig(f"figs/SVG/SpearmanCorrelation.svg",dpi = 300,pad_inches = 0.0)
        plt.savefig(f"figs/PDF/SpearmanCorrelation.pdf",dpi = 300,pad_inches = 0.0)
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

    # Load the data
    selection = ("*_leo_simplePath_lidar_on",)
    data = np.ndarray((0, 21)) 
    for filepath in sorted(sum((glob.glob(f"**/test_data/{select}/**/*.json", recursive=True) for select in selection), start=[])):
        print(f"Loading {filepath}...")
        # data = np.vstack([data, training.flatten_data_to_array(data_path=filepath)])
        data = training.flatten_data_to_array(data_path=filepath)
    
        print(data.shape)
        # training.train(data, output=True, output_name="mag_nn_model_v4.1.8")
        training.eval(
            data,
            NNSize=20,
            model_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/models/mag_nn_model_v4.1.8.pth",
        )
    # training.plot_correlation(data)


if __name__ == "__main__":
    main()

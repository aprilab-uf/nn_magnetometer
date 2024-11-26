import matplotlib.pyplot as plt
import json
import numpy as np
from magnav_nn import TrainData


def flatten_data_to_array(data):
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

        # Create a flattened list of values
        flattened_entry = [
            entry["true_magnetic_magnitude"],  # Magnetic magnitude
            entry["nn_magnetic_magnitude"],
            entry["map_magnetic_magnitude"],
        ]
        flattened_data.append(flattened_entry)

    # Convert the flattened data list to a NumPy array
    flattened_data_np = np.array(flattened_data)
    return flattened_data_np

def main():
    train = TrainData()
    data = train.flatten_data_to_array("/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V3_Training/221124T142204_training_data_leo_lawnmowerSplitY_continous_lidar_off/leo_data.json")
    train.eval(data,NNSize=20,model_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/models/mag_nn_model_v4.1.8.pth", save_data= True)
    # train.eval(data,"/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/models/mag_nn_model_v3.2.pth")

if __name__ == "__main__":
    main()

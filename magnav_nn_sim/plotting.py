import matplotlib.pyplot as plt
import json
import numpy as np
from magnav_nn import TrainData

def main():
    train = TrainData()
    data = train.flatten_data_to_array("/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/training_data/V3_Training/221124T142204_training_data_leo_lawnmowerSplitY_continous_lidar_off/leo_data.json")
    train.eval(data,NNSize=20,model_path="/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/models/mag_nn_model_v4.1.8.pth", save_data= True)
    # train.eval(data,"/home/basestation/magnav_sim_ws/src/magnav_nn_sim/data/models/mag_nn_model_v3.2.pth")

if __name__ == "__main__":
    main()

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from statsmodels.nonparametric.smoothers_lowess import lowess


def prep_data(filename):
    """
    Read data, 
    Take only time, x, and y
    Deal with 
    """
    column_name = ['time', 'x_accel', 'y_accel', 'z_accel', 'total_accel']
    raw_data = pd.read_csv(filename, names=column_name, header=None) 
    raw_data = raw_data.drop(0)

    raw_data['time'] = raw_data['time'].apply(to_float)
    raw_data['total_accel'] = raw_data['total_accel'].apply(to_float)
    raw_data['x_accel'] = raw_data['x_accel'].apply(to_float)
    raw_data['y_accel'] = raw_data['y_accel'].apply(to_float)
    raw_data['z_accel'] = raw_data['z_accel'].apply(to_float)
    
    accel = raw_data['total_accel'].tolist()
    time = raw_data['time'].tolist()
    
    filtered = lowess(accel, time, frac=0.05)
    plt.plot(time, accel, 'b.', linewidth=1, alpha=0.5)
    plt.plot(filtered[:,0], filtered[:,1], 'r-', linewidth=2)
    plt.show()
    #plt.plot(time, accel, 'b.', alpha=0.2)
    
    """
    # Kalman Filter
    std = np.std(accel)
    print(std)
    initial_guess = [0]
    observation_covariance = std**2#np.diag([std, std]) ** 2
    kf = KalmanFilter(
            initial_state_mean= initial_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            )
    pred_state, state_cov = kf.smooth(accel)
    
    plt.plot(time, pred_state[:,0], 'g-', linewidth=2, alpha=0.5)
    plt.plot(time, accel, 'b.', alpha = 0.1)
    plt.show()
    """   
    
def calc_distance(x, y):
    vector = np.subtract(x, y)
    return np.linalg.norm(vector)  

def to_float(x):
    return float(x)
    
def main():
    prep_data("data/alex-hand30s.csv")

    
if __name__ == "__main__":
    main()
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from statsmodels.nonparametric.smoothers_lowess import lowess


def prep_data(filename):
    """
    Read file and convet
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
    x = raw_data['x_accel'].tolist()
    y = raw_data['y_accel'].tolist()
    z = raw_data['z_accel'].tolist()

    # Lowess accelerations    
    x_lowess = lowess(x, time, frac=0.09)
    y_lowess = lowess(y, time, frac=0.09)
    z_lowess = lowess(z, time, frac=0.09)

    x_vel = np.trapz(x_lowess[:,1], time)
    y_vel = np.trapz(y_lowess[:,1], time)
    z_vel = np.trapz(z_lowess[:,1], time)
    
    print(x_vel, y_vel, z_vel)
    print(calc_distance((0,0), (x_vel, y_vel)))    
    lowess_columns = {'time':time,
                      'x_lowess':x_lowess[:,1],
                      'y_lowess':y_lowess[:,1],
                      'z_lowess':z_lowess[:,1]
                      }
    data_lowess = pd.DataFrame(lowess_columns)
    plt.plot(time, x_lowess[:,1], "r-", linewidth=2, alpha=0.2)
    plt.plot(time, y_lowess[:,1], "g-", linewidth=2, alpha=0.2)
    plt.plot(time, z_lowess[:,1], "b-", linewidth=2, alpha=0.2)
    plt.show()
    
    # Kalman Filter accelerations
    
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
    #prep_data("data/p2-pocket-1.csv")
    prep_data("data/p1-ankle-1.csv")

    
if __name__ == "__main__":
    main()
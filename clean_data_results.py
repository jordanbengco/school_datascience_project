import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from statsmodels.nonparametric.smoothers_lowess import lowess

def to_float(x):
    return float(x)

def get_csv(filename):
    column_name = ['time', 'x_accel', 'y_accel', 'z_accel', 'total_accel']
    raw_data = pd.read_csv(filename, names=column_name, header=None) 
    raw_data = raw_data.drop(0) 
    raw_data['time'] = raw_data['time'].apply(to_float)
    raw_data['x_accel'] = raw_data['x_accel'].apply(to_float)
    raw_data['y_accel'] = raw_data['y_accel'].apply(to_float)
    raw_data['z_accel'] = raw_data['z_accel'].apply(to_float)
    
    return raw_data

def main():
    raw_data1 = get_csv("data/p2-ankle-1.csv")
    raw_data2 = get_csv("data/p2-ankle-2.csv")
    raw_data3 = get_csv("data/p2-ankle-3.csv")
    raw_data4 = get_csv("data/p2-hand-1.csv")
    raw_data5 = get_csv("data/p2-hand-2.csv")
    raw_data6 = get_csv("data/p2-hand-3.csv")
    raw_data7 = get_csv("data/p2-pocket-1.csv")
    raw_data8 = get_csv("data/p2-pocket-2.csv")
    raw_data9 = get_csv("data/p2-pocket-3.csv")

    raw_data1 = raw_data1[raw_data1['time'] < 30]
    raw_data2 = raw_data2[raw_data2['time'] < 30]
    raw_data3 = raw_data3[raw_data3['time'] < 30]
    raw_data4 = raw_data4[raw_data4['time'] < 30]
    raw_data5 = raw_data5[raw_data5['time'] < 30]
    raw_data6 = raw_data6[raw_data6['time'] < 30]
    raw_data7 = raw_data7[raw_data7['time'] < 30]
    raw_data8 = raw_data8[raw_data8['time'] < 30]
    raw_data9 = raw_data9[raw_data9['time'] < 30]
  
    mean1 = raw_data1.std().tolist()
    mean2 = raw_data2.std().tolist()
    mean3 = raw_data3.std().tolist()
    mean4 = raw_data4.std().tolist()
    mean5 = raw_data5.std().tolist()
    mean6 = raw_data6.std().tolist()
    mean7 = raw_data7.std().tolist()
    mean8 = raw_data8.std().tolist()
    mean9 = raw_data9.std().tolist()
  
    frames = [mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9]
    col = ['time', 'x_accel', 'y_accel', 'z_accel']
    results = pd.DataFrame(frames, columns=col)
    results['tester'] = 2
    print(results)
    results.to_csv("p2-std.csv")    

if __name__ == "__main__":
    main()
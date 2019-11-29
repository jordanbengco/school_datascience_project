import sys
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

def df_to_list(dataframe):
    ## Outputs to three lists ankle, hand, pocket and the tester 
    ## Assumed: The dataframe is for the same tester and it is ordered by 3 ankles, 3 hands, 3 pockets
    total_list = dataframe[['x_accel', 'y_accel', 'z_accel']].values.tolist()
    
    ankle = total_list[0:3]
    hand = total_list[3:6]
    pocket = total_list[6:9]
    tester = dataframe['tester'].tolist()[0:3] # array of size 3 so easy to plug into GaussianNB
    
    return ankle, hand, pocket, tester
    
def full_df_to_list(dataframe):
    ## Outputs full list regardless of where data was recorded
    ## Assumed: The dataframe is for the same tester
    total_list = dataframe[['x_accel', 'y_accel', 'z_accel']].values.tolist()
    tester = dataframe['tester'].tolist()
    
    return total_list, tester

def gaussian():
    ## TODO
    ## Run data through bayesian classifier? Other ML classifiers?
    
def main():
    p1_data_mean = pd.read_csv('p1-mean.csv')
    p1_data_std = pd.read_csv('p1-std.csv')
    p2_data_mean = pd.read_csv('p2-mean.csv')
    p2_data_std = pd.read_csv('p2-std.csv')
    
    df_to_list(p1_data_mean)
    
    
if __name__ == "__main__":
    main()
# sensor data project

# Project for cmpt 353.

# Required Libraries:
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pykalman import KalmanFilter
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from statsmodels.nonparametric.smoothers_lowess import lowess


# Order of Execution:
We need to run both the py and ipynb since, even though its the same app, collects data differently on a Samsung device then an Apple device and the two programs clean data one for each.
1. ./clean_data_results.py
2. Run all clean_data_results.ipynb
3. Run all clean_data_speed.ipynb
4. ./clean_data_speed.py
5. Run all results_test.ipynb


# Report:
- The problem you are addressing, particularly how you refined the provided data
- The data that you use: how it was gathered, cleared, etc.
- Techniques you used to analyse the data.
- Your results/findings/conclusions
- Some appropriate visualizations of your data/results
- Limitations: problems you encountered, things you would do if you had more time, things you should have done it retrospective.
#%%
# Headers
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import random
import os

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

# %%
# Fetching the data
datadir = '../Data'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
# Reading data from csv using pandas 
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names=columns)
# Setting unlimited column width
pd.set_option('display.max_colwidth', None)
data.head()

# Stripping file name from the path
def path_leaf(path):
    return os.path.basename(path)

# Apply the function to all selected columns simultaneously
for col in ['center', 'left', 'right']:
    data[col] = data[col].apply(path_leaf)
data.head()

# %%
# Plotting steering angle for visualization in histogram
# For identifying the steering angle which is most common
num_bins = 25
# Getting histogram and bins
hist, bins = np.histogram(data['steering'], num_bins)
print(bins)
# %%

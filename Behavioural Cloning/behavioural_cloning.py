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

# %%

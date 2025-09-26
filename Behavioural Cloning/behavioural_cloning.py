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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

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
# Most samples are around 0 angle
# Hence adding a threashold of 200 to uniformize the data
samples_per_bin = 350
# Getting histogram and bins
hist, bins = np.histogram(data['steering'], num_bins)
# Centering the value around 0
center = (bins[:-1]+bins[1:]) * 0.5
# Plotting the steering angle 
plt.bar(center, hist, width=0.05)
# Plotting cut off for more than 200 samples
plt.plot([np.min(data['steering']), np.max(data['steering'])], (samples_per_bin, samples_per_bin))

# %%
# Remove classes which are above threshold of 350
print('Total data : ', len(data))
remove_list = []
for j in range (num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        # Removing similar steering angles
        if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
            list_.append(i)
            # Shuffle the data and remove the last 300 data so that full track data is available after removing
    list_ = shuffle(list_)
    # Slicing the list
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)

print('Removed : ', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('Remaining: ', len(data))
hist, _= np.histogram(data['steering'], num_bins)
# Plotting the steering angle 
plt.bar(center, hist, width=0.05)
# Plotting cut off for more than 200 samples
plt.plot([np.min(data['steering']), np.max(data['steering'])], (samples_per_bin, samples_per_bin))

# %%
# Creating training and validation data
def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(data)):
        # iloc - allows to perform a selection on row of data based on index
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        # Appending the main path to the image name
        image_path.append(os.path.join(datadir, center.strip()))
        # Append the steering angles
        steering.append(float(indexed_data[3]))
    image_path = np.asarray(image_path)
    steering = np.asarray(steering)
    return image_path, steering

image_paths, steerings = load_img_steering(datadir + '/IMG', data)


# %%
# Splitting the test and validation data
X_train, X_val, y_train, y_val = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
print(f'Training samples: {len(X_train)}\nValidation Samples: {len(X_val)}')
# Ensuring that both is having data distribution uniform
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_train, bins=num_bins, width=0.05, color='b')
axes[0].set_title('Training Set') 
axes[1].hist(y_val, bins=num_bins, width=0.05, color='r')
axes[1].set_title('Validation Set') 
# %%

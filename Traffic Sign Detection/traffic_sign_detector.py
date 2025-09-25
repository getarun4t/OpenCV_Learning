#%%
# Headers
import numpy as np
import matplotlib.pyplot as plt
import keras
# For unpickling the data
import pickle
# Importing data analysis library
# Used for manipulating and analysing data inside the csv file
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# %%
# Seed data
np.random.seed(0)
path = "../../german-traffic-signs"

#%%
# Unpickling the data and opening the files
# German traffic signs bitbucket repo cloned for this purpose
with open(f'{path}/train.p', 'rb') as f:
    train_data = pickle.load(f)
with open(f'{path}/valid.p', 'rb') as f:
    val_data = pickle.load(f)
with open(f'{path}/test.p', 'rb') as f:
    test_data = pickle.load(f)

print(type(train_data))

X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = val_data['features'], val_data['labels']
X_test, y_test = test_data['features'], test_data['labels']

# Inspecting the shape of the data
# Depth of 3, hence RGB Image
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

# Verifying that number of images is equal to number of labels
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to number of labels"
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to number of labels"
assert(X_val.shape[0] == y_val.shape[0]), "The number of images is not equal to number of labels"

# Verifying that dimensions of the images
assert(X_train.shape[1:] == (32, 32, 3)), "The dimension of images is not 32, 32, 3"
assert(X_test.shape[1:] == (32, 32, 3)), "The dimension of images is not 32, 32, 3"
assert(X_val.shape[1:] == (32, 32, 3)), "The dimension of images is not 32, 32, 3"

# %%
# Reading the CSV file using pandas
data = pd.read_csv(f'{path}/signnames.csv')
print(data)

# %%

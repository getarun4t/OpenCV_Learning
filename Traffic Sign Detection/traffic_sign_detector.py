#%%
# Headers
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
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
#Creating variables for plotting to visualize the data
num_of_samples = []

# To visualize random set of 5 images 
cols = 5
# 0-9 images
num_of_classes = 43

fig, axis = plt.subplots(nrows=num_of_classes, ncols=cols, figsize=(5, 50))
fig.tight_layout()

#fill the cells
for i in range(cols):
    # Iterate over the entire data
    for j, row in data.iterrows():
        x_selected = X_train[y_train==j]
        # Access random image from jth category for the cell
        # Ensuring grey scale
        axis[j][i].imshow(x_selected[random.randint(0, len(x_selected)-1)],cmap=plt.get_cmap("grey"))
        axis[j][i].axis("off")
        if i == 2:
            axis[j][i].set_title(str(j) + "-" + row['SignName'])
            num_of_samples.append(len(x_selected))

# %%
# Visualize class distribution 
# Data set distribution is different among classes
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_of_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")

#%%
# Plotting a random original image first
print("Unprocessed image")
plt.imshow(X_train[1000])
plt.axis("off")
print(X_train[1000].shape)
print(y_train[1000])

# %%
# Preprocessing the image
# 1. Converting to greyscale
def greyscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

img = greyscale(X_train[1000])
plt.imshow(img, )
plt.axis("off")
print(img.shape)

# %%
# 2. Histogram equalization
# Standardizes lighting in all images
# Ensure images have similar brighting value
def hist_equalize(img):
    # Function only extracts greyscale images
    img = cv2.equalizeHist(img)
    return img

img = hist_equalize(img)
plt.imshow(img)
plt.axis("off")
print(img.shape)

# %%
# Preprocessing all the images
def preprocessing(img):
    img = greyscale(img)
    img = hist_equalize(img)
    # Normalizing the image to pixel value 255
    img = img/255
    return img

# Iterating through entire list of 
X_train = np.array(list(map(preprocessing, X_train)))
X_test = np.array(list(map(preprocessing, X_test)))
X_val = np.array(list(map(preprocessing, X_val)))

# %%
# Randomly plotting a processed image
# cmap=plt.cm.binary used to force imshow to plot greyscale
plt.imshow(X_train[random.randint(0, len(X_train)-1)], cmap=plt.cm.binary)
plt.axis("off")
print(X_train.shape)
# %%

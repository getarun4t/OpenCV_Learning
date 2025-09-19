# %%
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# Importing mnist data set
from keras.datasets import mnist
from keras.utils import to_categorical
import random

# %%
# Ensures same random input always
np.random.seed(0)

# %%
# Training data set and test data set loading
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

# %%
# Verifying the load
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to number of labels"
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to number of labels"
assert(X_train.shape[1:] == (28,28)), "The dimension of images is not 28*28"
assert(X_test.shape[1:] == (28,28)), "The dimension of images is not 28*28"

# %%
#Creating variables for plotting to visualize the data
num_of_samples = []

# To visualize random set of 5 images 
cols = 5
# 0-9 images
num_of_classes = 10

fig, axis = plt.subplots(nrows=num_of_classes, ncols=cols, figsize=(5, 10))
fig.tight_layout()

#fill the cells
for i in range(cols):
    for j in range (num_of_classes):
        x_selected = X_train[y_train==j]
        # Access random image from jth category for the cell
        # Ensuring grey scale
        axis[j][i].imshow(x_selected[random.randint(0, len(x_selected)-1)],cmap=plt.get_cmap("grey"))
        axis[j][i].axis("off")
        if i == 2:
            axis[j][i].set_title(str(j))
            num_of_samples.append(len(x_selected))

# %%
#Visualize class distribution 
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_of_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")

# %%
# One hot encoding
# Ensures that independent label for each class
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
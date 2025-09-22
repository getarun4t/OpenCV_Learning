# %%
# Header files
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# Importing mnist data set
from keras.datasets import mnist
from keras.utils import to_categorical
# Get Image module from Python Image Library
from PIL import Image
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

#%%
#Normalizing the data
# Max pixel intensity of 255 gets normalized to b/w 0 and 1
# Decreases variance b/w data
X_train = X_train/255
X_test = X_test/255

#%%
# Define the pixels
num_pixels = 784
X_train = X_train.reshape(X_train.shape[0], num_pixels)
print(X_train.shape)
X_test = X_test.reshape(X_test.shape[0], num_pixels)
print(X_test.shape)


# %%
#Implementing with Regular Deep Neural Network
def create_model():
    model = Sequential()
    # More hidden layers will result in overfitting
    # Relu activation function is non-linear
    # Relu performs better for convolution
    model.add(Dense(10, input_dim = num_pixels, activation="relu"))
    model.add(Dense(10, activation='relu'))
    # Softmax converts all scores to probabilities
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#%%
# Calling and printing the model
# First layer has 7850 bias and weights
# Computational power required while using DNN is very high for larger images
# Hence we need to use convolutional neural network
model = create_model()
print(model.summary())

# %%
# Training data
# All labels are in y_train
history = model.fit(X_train, y_train, validation_split=0.1, epochs = 10, batch_size=200, verbose=1, shuffle=1)

# %%
# Plotting the loss data
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.title('Loss')
plt.xlabel('epoch')

# %%
# Plotting the accuracy data
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
plt.title('Accuracy')
plt.xlabel('epoch')

# %%
# Testing the model
score = model.evaluate(X_test, y_test, verbose=0)
print(type(score))
print('Test Score of Deep Neural Networks: ', score[0])
print('Test Accruacy of Deep Neural Networks: ', score[1])
# %%
# Test with image from web
# Getting the image from web
url = "https://colah.github.io/posts/2014-10-Visualizing-MNIST/img/mnist_pca/MNIST-p1815-4.png"
response = requests.get(url, stream=True)
print(response)
#Getting the raw image
img = Image.open(response.raw)
plt.imshow(img)

# %%
# Image handling
# Getting and resizing 28 * 28 image array
img_array = np.asarray(img)
print(img_array.shape)
# Resizing to 28*28
img_resized = cv2.resize(img_array, (28, 28))
print(img_resized.shape)
# Changing image to greyscale
img_grey_scale = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
plt.imshow(img_grey_scale, cmap=plt.get_cmap("grey"))
# Inverting the image to white image and black background
image = cv2.bitwise_not(img_grey_scale)
plt.imshow(image, cmap=plt.get_cmap("grey"))

# %%
# Normalizing the image
image = image/255
image = image.reshape(1, 784)
print(image)

# %%
#Predicting the image
predictions = model.predict(image)
prediction_class = np.argmax(predictions, axis=1)
print("Prediction: ", str(prediction_class))

# If prediction is not correct, try adding depth
# More efficient approach, use convolutional neural network
# %%

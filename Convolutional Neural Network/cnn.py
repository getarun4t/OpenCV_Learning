#%%
# Starter code
import numpy as np
import matplotlib.pyplot as plt
import requests
import cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
# Adding headers for CNN
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
# Get Image module from Python Image Library
from PIL import Image
import random


np.random.seed(0)




(X_train, y_train), (X_test, y_test)= mnist.load_data()
 
print(X_train.shape)
print(X_test.shape)
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (28,28)), "The dimensions of the images are not 28 x 28."
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
assert(X_test.shape[1:] == (28,28)), "The dimensions of the images are not 28 x 28."
 
num_of_samples=[]
 
cols = 5
num_classes = 10
 
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,10))
fig.tight_layout()
 
for i in range(cols):
    for j in range(num_classes):
      x_selected = X_train[y_train == j]
      axs[j][i].imshow(x_selected[random.randint(0,(len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
      axs[j][i].axis("off")
      if i == 2:
        axs[j][i].set_title(str(j))
        num_of_samples.append(len(x_selected))




print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the train dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()
 

#%%
# Adding depth to the data
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
 
X_train = X_train/255
X_test = X_test/255

# %%
# Define LeNet Model
def leNet_model():
   model = Sequential()
   # Convolutional Layer
   # 30 filters is good
   # 5,5 is filter matrix size
   # 28, 28, 1 is filter shape (3d matrix)
   # strides - translation of the kernel
   # padding - preserves spatial size of i/p
   # padding ensures o/p size same as i/p, to be used only if outer edges of image is imp
   model.add(Conv2D(30, (5, 5), input_shape = (28, 28, 1), activation='relu'))
   # Pooling layer
   # size is scaled down by half
   model.add(MaxPooling2D(pool_size=(2, 2)))
   # Second Conv layer
   # Smaller filter because image is now smaller
   # No input_shape as it is not the first layer
   # Each image scaled down to 10,10,15
   # Depth increases but image size reduces
   model.add(Conv2D(15, (3,3), activation='relu'))
   # Second pooling layer
   # Reduces size to 5,5,15
   model.add(MaxPooling2D(pool_size=(2, 2)))
   # Flattening the image
   # Doesn't require any param
   # Image has to be 1D before adding to perceptrons
   # Output is a 1D array of shape 375
   model.add(Flatten())
   # Fully Connected Layer
   # Dense layer
   model.add(Dense(500, activation='relu'))
   # Output layer
   # Activation is softmax so as to classify between different classes
   model.add(Dense(num_classes, activation='softmax'))
   # Optimizer
   model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
   return model

#%%
# Running the model
model = leNet_model()
# Train the model
# shuffle = 1 ensures data is shuffled during training
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=400, verbose=1, shuffle=1)

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
image = image.reshape(1, 28, 28, 1)
print(image)

# %%
#Predicting the image
predictions = model.predict(image)
prediction_class = np.argmax(predictions, axis=1)
print("Prediction: ", str(prediction_class))

# %%
# Checking accuracy
score = model.evaluate(X_test, y_test, verbose=0)
print(type(score))
print('Test Score of Convolutional Neural Networks: ', score[0])
print('Test Accruacy of Convolutional Neural Networks: ', score[1])
# %%

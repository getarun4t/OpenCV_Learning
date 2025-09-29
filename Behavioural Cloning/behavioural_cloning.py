#%%
# Headers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pandas as pd
import random
import os

#Image augmentor
from imgaug import augmenters as iaa

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

#%%
# Zooming the image
def zoom(image):
    # 1, 1,3 is range of zoom
    zoom=iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)
    return image

image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
zoomed_image = zoom(original_image)
# Plotting
fig, axis = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axis[0].imshow(zoomed_image)
axis[0].set_title('Zoomed image')
axis[1].imshow(original_image)
axis[1].set_title('Original image')

#%%
# pan image augmentation
def pan(image):
    pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image

image = image_paths[random.randint(0, 2001)]
original_image = mpimg.imread(image)
panned_image = pan(original_image)
# Plotting
fig, axis = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axis[0].imshow(panned_image)
axis[0].set_title('Panned image')
axis[1].imshow(original_image)
axis[1].set_title('Original image')

#%%
# Brightness change augmentation
def img_random_brightness(image):
    # Model reacts better to a higher fraction of darker images
    brightness = iaa.Multiply((0.2, 1.2))
    return brightness.augment_image(image)

image = image_paths[random.randint(0, 3939)]
original_image = mpimg.imread(image)
brightened_image = img_random_brightness(original_image)
# Plotting
fig, axis = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axis[0].imshow(brightened_image)
axis[0].set_title('Brightness altered image')
axis[1].imshow(original_image)
axis[1].set_title('Original image')

#%%
# Flipping image augmentation
def flip_image(image, steering_angle):
    # second arg is type of flip
    # 1 is horizontal flip
    flip = cv2.flip(image, 1)

# %%
# Preprocessing data
def image_preprocess(img):
    # Getting actual image from image path
    img = mpimg.imread(img)
    # Cropping the top and bottom of image with unnecessary data
    # Decided based on viewing the data
    img = img[60:135, :, : ]
    # Nvidia model is used
    # YUV color space required for handling
    # Y - Brightness, UV - Cromiance (adds colors)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Adding Gaussian blur
    # Smoothens image
    img = cv2.GaussianBlur(img, (3,3), 0)
    # Resize image for faster handling
    # Input size of Nvidia architecture (for consistency)
    img = cv2.resize(img, (200, 66))
    # Normalization
    # No visual impact
    img = img/255
    return img

image = image_paths[100]
original_image = mpimg.imread(image)
preprocessed_image = image_preprocess(image)

#Plotting both images for comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axes[0].imshow(original_image)
axes[0].set_title('Original Image')
axes[1].imshow(preprocessed_image)
axes[1].set_title('PreProcessed Image')

# Preprocessing all data
X_train = np.array(list(map(image_preprocess, X_train)))
X_val = np.array(list(map(image_preprocess, X_val)))
# Checking a random image
plt.imshow(X_train[random.randint(0, len(X_train-1))])
plt.axis('off')
print(X_train.shape)

# %%
# Defining the NVidia Model
# Behavioural cloning data set is regression type as it should return steering angle
# 200*66 size
# 1000 + data classes
def nvidia_model():
    # Initializing the Nvidia model (normalizing data is already done)
    model = Sequential()
    # Convolutional layer
    # subsample = stride length of the image, can be increased for quicker computation
    # padding not added as edges are not imp
    # use of relu causes DEAD relu when value is less than zero
    # negative input is always fed forward and will be always zero
    # instead use elu instead of relu
    # Elu has non-zero value in negative
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu', input_shape=(66, 200, 3)))
    # Convolutional layer 2
    # Should have 36 filters with kernal size 5, 5
    model.add(Conv2D( 36, (5, 5), strides=(2,2), activation='elu'))    
    # Convolutional layer 3
    # Should have 36 filters with kernal size 5, 5
    model.add(Conv2D( 48, (5, 5), strides=(2,2), activation='elu'))    
    # Convolutional layer 4
    # Should have 36 filters with kernal size 5, 5
    # subsampling is removed as image size is reduced drastically after first 3 layers
    model.add(Conv2D( 64, (3, 3), activation='elu'))  
    # Convolutional layer 5
    # Should have 36 filters with kernal size 5, 5
    model.add(Conv2D( 64, (3, 3), activation='elu'))

    # Seperate Convulational Layer from Fully Connected Layer
    model.add(Dropout(0.5))
    
    # Flatter layer
    # Do maths to find the dimensions of output
    model.add(Flatten())

    # 3 Dense layers
    model.add(Dense(100, activation='elu'))
    
    # Prevent Overfitting
    # Adding a dropout layer 
    model.add(Dropout(0.5))
    
    model.add(Dense(50, activation='elu'))

    # Prevent Overfitting
    # Adding a dropout layer
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='elu'))

    # Prevent Overfitting
    # Adding a dropout layer
    model.add(Dropout(0.5))

    # Dense layer with single output node
    model.add(Dense(1))

    # Compiling the architecture
    # mean square error used for error metric
    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss = 'mse', optimizer=optimizer)

    return model

#%%
# Defining the model    
model = nvidia_model()
print(model.summary())

# %%
# Training the model
# More epochs since data is less
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=100, verbose=1, shuffle=1)

#%%
# Plotting the loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epochs')

# %%
# Saving the model
model.save('model_tf.keras')  
print(os.getcwd())

# %%

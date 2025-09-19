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

# %%

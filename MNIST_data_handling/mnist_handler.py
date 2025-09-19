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
np.random.seed(0)
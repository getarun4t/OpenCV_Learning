# %%
import numpy as np
import matplotlib.pyplot as plt
# for importing data sets which are complex (non-linearly seperable)
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# %%
np.random.seed(0)

# %%
# number of data points, each data point will have a label
n_pts = 500
# optimum noise value which is not so low
# high noise will make data points overly convoluted
# lower factor value for seperation btw data sets
X, y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)
# %%
# Plotting outer region
plt.scatter(X[y==0, 0], X[y==0, 1])
# Plotting inner region
plt.scatter(X[y==1, 0], X[y==1, 1])

# %%
# Creating a deep neural network
# Creating an input layer with 2 nodes
# Creating a hidden layer with 4 node
# Creating an output layer with one node
model = Sequential()
model.add(Dense(4, input_shape=(2,), activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# Compiling the model using Adam optimizer
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# %%
# Train a model to fit our data
# verbose ensures that feedback in screen is given
# epoch defines how many datapoints are handled in 1 iteration
# shuffle ensures that model doesn't get stuck in absolute minimum
model.fit(x=X, y=y, verbose=1, batch_size=20, epochs=100, shuffle='true')
# %%

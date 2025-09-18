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
h = model.fit(x=X, y=y, verbose=1, batch_size=20, epochs=100, shuffle='true')
# %%
# Plotting the accuracy values
plt.plot(h.history['accuracy'])
plt.xlabel('epoch')
plt.legend(['accuracy'])
plt.title('accuracy')
# %%
# Plotting the loss values
plt.plot(h.history['loss'])
plt.xlabel('epoch')
plt.legend(['loss'])
plt.title('loss')
# %%
# Decision boundary func
def plot_decision_boundary(X, Y, model):
    x_span = np.linspace(min(X[:, 0]) - 0.25, max(X[:, 0]) + 0.25)
    y_span = np.linspace(min(X[:, 1]) - 0.25, max(X[:, 1]) + 0.25)
    # square 2d array in numpy mesh grid func
    # returns 2 2d 50*50 matrix for each span 
    xx, yy = np.meshgrid(x_span, y_span)
    # converting to 1d because currently every y coordinate has 50 x coordinates
    # so converting to columnwise to concantenate
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    # returns array of predictions with probability of returning 1 or 0
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    # plotting contour, represents probability as contours 
    plt.contourf(xx, yy, z)

# %%
# Plotting the decision boundary
plot_decision_boundary(X, y, model)
# Plotting the data
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])
# %%
# Predicting the value
plot_decision_boundary(X, y, model)
# Plotting the data
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])
x= 0.1
y = 0
point = np.array([[x,y]])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize = 10, color = 'red')
print("Prediction is : ", prediction)
# %%

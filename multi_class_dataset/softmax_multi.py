# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
# %%
# Setting number of data points
n_pts = 500
centers = [[-1, 1], [-1, -1], [1, -1]]
# random_state allows to get same random data on rerun
X, y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std= 0.4)
print(X)
print(y)
# %%
# plotting
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
# %%
# Hot encoding-> eliminates unnecessary relations b/w datasets
# 3 is number of data classes
y_cat = to_categorical(y, 3)
print("y= " , y_cat)

# %%
model = Sequential()
model.add(Dense(units=3, input_shape =(2,), activation="softmax"))
model.compile(Adam(0.1), loss="categorical_crossentropy", metrics=['accuracy'])

# %%
model.fit(x=X, y=y_cat, verbose=1, batch_size=50, epochs=100)
# %%
# Decision boundary func
def plot_decision_boundary(X, y_cat, model):
    x_span = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1)
    y_span = np.linspace(min(X[:, 1]) - 1, max(X[:, 1]) + 1)
    # square 2d array in numpy mesh grid func
    # returns 2 2d 50*50 matrix for each span 
    xx, yy = np.meshgrid(x_span, y_span)
    # converting to 1d because currently every y coordinate has 50 x coordinates
    # so converting to columnwise to concantenate
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    # returns array of predictions with probability of returning 1 or 0
    # predict_classes specifically for multi class
    # Instead of predict_classes (deprecated), use predict + argmax
    pred_probs = model.predict(grid)
    pred_func = np.argmax(pred_probs, axis=1)
    z = pred_func.reshape(xx.shape)
    # plotting contour, represents probability as contours 
    plt.contourf(xx, yy, z)

# %%
# Plotting the decision boundary
plot_decision_boundary(X, y_cat, model)
# Plotting the data
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
# %%
# Replotting with test input
plot_decision_boundary(X, y_cat, model)
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
x = 0.5
y = 0.5
point = np.array([[x, y]])
prediction = np.argmax(model.predict(point), axis=1)
plt.plot([x], [y], marker = 'o', markersize=10, color = 'b')
print("Prediction is ", prediction)
# %%

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
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
# Hot encoding, eliminates unnecessary relations b/w datasets
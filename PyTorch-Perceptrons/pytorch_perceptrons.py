# %%
# Headers
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

#%%
# Creating datasets
n_pts = 100
# Defining central coordinates for our cluster
centers = [[-0.5, 0.5], [0.5, -0.5]]
# Storing data points and labels corresponding to it
# Cluster_std - standard deviation from center point
X, y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4)
print(X)
print(y)

#%%
#Plotting the data points and labels
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])

#%%
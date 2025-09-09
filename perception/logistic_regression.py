# %%
import numpy as np
import matplotlib.pyplot as plt

#Data size
n_pts = 100
np.random.seed(0)  #Ensures same set of data generated each time

# Creating random data as array points
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts)]).transpose()
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts)]).T
_, ax = plt.subplots(figsize=(4,4))

#Adding color and scattering result
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')

#Plotting results
plt.show()
# %%

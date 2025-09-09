# %%
import numpy as np
import matplotlib.pyplot as plt


def draw(x1, x2):
    ln = plt.plot(x1, x2)

def sigmoid(score):
    return 1/(1+ np.exp(-score))

#Data size
n_pts = 100
np.random.seed(0)  #Ensures same set of data generated each time

#All data is having a bias with value 1
bias = np.ones(n_pts)

# Creating random data as array points
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).transpose()
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T


#Stacking the points
all_points = np.vstack((top_region, bottom_region))

#Random weights and bias
w1 = -0.2
w2 = -0.35
b = 3.5

#Finding the points
line_parameters = np.matrix([w1, w2, b]).T
x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
x2 = -(w1/w2) * x1 - (b/w2)  # since  w1x1+w2x2=0
linear_combination = all_points * line_parameters
probabilites = sigmoid(linear_combination)

_, ax = plt.subplots(figsize=(4,4))
#Adding color and scattering result
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
draw(x1, x2)

#Plotting results
plt.show()
# %%

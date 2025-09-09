# %%
import numpy as np
import matplotlib.pyplot as plt


def draw(x1, x2):
    ln = plt.plot(x1, x2)

def sigmoid(score):
    return 1/(1+ np.exp(-score))

def calculate_error(line_parameters, points, y):
    """
    Finding cross entropy value
    """
    m = points.shape[0]
    scores = points @ line_parameters    # matrix multiply
    probabilities = sigmoid(scores)      # (20,1)
    cross_entropy = -(1/m) * np.sum(
                        y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities)
                        )
    return cross_entropy

#Data size
n_pts = 10
np.random.seed(0)  #Ensures same set of data generated each time

#All data is having a bias with value 1
bias = np.ones(n_pts)

# Creating random data as array points
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).transpose()
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T


#Stacking the points
all_points = np.vstack((top_region, bottom_region))

#Random weights and bias
w1 = -0.1
w2 = -0.15
b = 0.5

#Finding the points
line_parameters = np.array([w1, w2, b]).reshape(-1, 1)
x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
x2 = -(w1/w2) * x1 - (b/w2)  # since  w1x1+w2x2=0
y = np.array([np.zeros(n_pts), np.zeros(n_pts)]).reshape(n_pts*2, 1)

_, ax = plt.subplots(figsize=(4,4))
#Adding color and scattering result
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
draw(x1, x2)

#Plotting results
plt.show()
# %%
print(calculate_error(line_parameters, all_points, y))

# %%

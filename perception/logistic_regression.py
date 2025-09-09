# %%
import numpy as np
import matplotlib.pyplot as plt


def draw(x1, x2):
    ln = plt.plot(x1, x2, '-')
    # plt.pause(0.0001) # Animated mode, not working
    # ln[0].remove()

def sigmoid(score):
    return 1/(1+ np.exp(-score))

def calculate_error(line_parameters, points, y):
    """
    Finding cross entropy value
    """
    n=points.shape[0]
    p= sigmoid(points*line_parameters)
    cross_entropy=-(1/n)*(np.log(p).T*y + np.log(1-p).T*(1-y))
    return cross_entropy

def gradient_descent(line_parameters, points, y, alpha):
    """
    alpha - learning rate
    """
    m = points.shape[0]
    for i in range(2000):
        p = sigmoid(points @ line_parameters)   # use @ for clarity
        gradient = (points.T @ (p - y)) * (1/m)
        line_parameters = line_parameters - alpha * gradient   # ✅ fixed update
        # For visualization
        w1 = line_parameters[0,0]
        w2 = line_parameters[1,0]
        b  = line_parameters[2,0]
        x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
        x2 = -(w1/w2) * x1 - (b/w2)
    draw(x1, x2)


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

#Finding the points
line_parameters = np.matrix([np.zeros(3)]).T  #making intial weights and bias as zero

y=np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)

_, ax = plt.subplots(figsize=(4,4))
#Adding color and scattering result
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')

#Performing gradient descent
gradient_descent(line_parameters, all_points, y, 0.06)

#Plotting results
plt.show()
# %%
print((calculate_error(line_parameters, all_points, y)))

# %%

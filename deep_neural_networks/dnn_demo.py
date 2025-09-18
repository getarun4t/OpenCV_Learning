# %%
import numpy as np
import matplotlib.pyplot as plt
# for importing data sets which are complex (non-linearly seperable)
from sklearn import datasets

# %%
np.random.seed(0)

# %%
# number of data points, each data point will have a label
n_pts = 500
# optimum noise value which is not so low
# high noise will make data points overly convoluted
# lower factor value for seperation btw data sets
X, y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)
print(X)
print(y)
# %%

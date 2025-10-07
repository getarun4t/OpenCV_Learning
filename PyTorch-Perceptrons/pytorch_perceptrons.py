# %%
# Headers
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from sklearn import datasets

#%%
# Creating datasets
n_pts = 100
# Defining central coordinates for our cluster
centers = [[-0.5, 0.5], [0.5, -0.5]]
# Storing data points and labels corresponding to it
# Cluster_std - standard deviation from center point
X, y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4)
X_data = torch.Tensor(X)
y_data = torch.Tensor(y.reshape(100,1))

#%%
#Plotting the data points and labels
def scatter_plot():
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
scatter_plot()

#%%
# Initializing the linear model
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Instance of class to be initialized
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, X):
        pred = torch.sigmoid(self.linear(X))
        return pred

#%%
# Testing the model with random seed
torch.manual_seed(2)
model = Model(2, 1)
print(list(model.parameters()))

#%%
# Obtain model parameters
[w, b] = model.parameters()
w1, w2 = w.view(2)
b1 = b[0]

def get_params():
    return (w1.item(), w2.item(), b[0].item())

#%%
# Plotting the parameters
def plot_fit(title):
    plt.title = title
    w1, w2, b1 = get_params()
    x1 = np.array([-2.0, 2.0])
    x2 = -(w1*x1 + b1) / w2
    plt.plot(x1, x2, 'r')
    scatter_plot()

plot_fit("Model")

#%%
# Only 2 classes, hence binary cross entropy
criterion = nn.BCELoss()
# After computing error, taking gradient and its direction
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
# Epoch - no. of times model pass over data set
epochs = 1000
losses = []
for i in range(epochs):
    y_pred = model.forward(X_data)
    # Entropy between predicted and actual
    loss = criterion(y_pred, y_data)
    print('Epcoh: ', i, " Loss: ", loss.item())
    
    losses.append(loss.item())
    optimizer.zero_grad()
    # Finding gradient of loss
    loss.backward()
    optimizer.step()

#%%
# %%
# Headers
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from sklearn import datasets

#%%
# Creating datasets
n_pts = 500
# Defining central coordinates for our cluster
centers = [[-0.5, 0.5], [0.5, -0.5]]
# Storing data points and labels corresponding to it
# noise lower value ensures that data is not too convoluted
X, y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)
X_data = torch.Tensor(X)
y_data = torch.Tensor(y.reshape(500,1))

#%%
#Plotting the data points and labels
def scatter_plot():
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
scatter_plot()

#%%
# Initializing the linear model
# H1 - Hidden layer
class Model(nn.Module):
    def __init__(self, input_size, H1, output_size):
        super().__init__()
        # Instance of class to be initialized
        self.linear = nn.Linear(input_size, H1)
        self.linear2 = nn.Linear(H1, output_size)

    def forward(self, X):
        X = torch.sigmoid(self.linear(X))
        X = torch.sigmoid(self.linear2(X))
        return X
    
    def predict(self, X):
        pred = self.forward(X)
        if pred >0.5:
            return 1
        else:
            return 0

#%%
# Testing the model with random seed
torch.manual_seed(2)
# Adding 4 hidden layers
# too little underfitting, too many overfitting
model = Model(2, 4, 1)
print(list(model.parameters()))

#%%
# Only 2 classes, hence binary cross entropy
criterion = nn.BCELoss()
# Using Adam optimizer
# It is using combination of 2 diff stachastic algorithms 
# lr is important for Adam
# Very efficient for large datasets
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
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
# Plotting the losses
plt.plot(range(epochs), losses)
plt.xlabel('Loss')
plt.ylabel('Epochs')

#%%
# Creating decision boundary
def plot_decision_boundary():
    # 0.25 tolerance 
    x_span = np.linspace(min(X[:, 0]) -0.25 , max(X[:, 0]) + 0.25)
    y_span = np.linspace(min(X[:, 1]) -0.25 , max(X[:, 1]) + 0.25)
    # Returning a 2d matrix (50*50)
    xx, yy = np.meshgrid(x_span, y_span)
    # Changing to 1D array and concantenate columnwise
    # np.c_ concantenize columnwise
    # Tensor - converts to tensor
    grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    # Feeding tensor to get predictions
    pred_func = model.forward(grid)
    # Reshaping pred to original data
    z = pred_func.view(xx.shape).detach().numpy()
    # Creating a contour plot
    plt.contourf(xx, yy, z)


# %%
# Plotting decision boundary
plot_decision_boundary()
scatter_plot()

# %%
# Testing value
x = 0.025
y = 0.025
point = torch.Tensor([x,y])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize = 10, color = 'r')
print("Prediction is", prediction)
plot_decision_boundary()
# %%

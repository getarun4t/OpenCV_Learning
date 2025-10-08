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
        self.linear = nn.Linear(input_size, output_size)
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
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
# Testing
point1 = torch.Tensor([1.0, -1.0])
point2 = torch.Tensor([-1.0, 1.0])
plt.plot(point1.numpy()[0], point1.numpy()[1], 'ro')
plt.plot(point2.numpy()[0], point2.numpy()[1], 'ko')

print(f"Red point in class: {model.predict(point1).item()}")
print(f"Black point class: {model.predict(point2).item()}")

print(f"Red point probability: {model.forward(point1).item()}")
print(f"Black point probability: {model.forward(point2).item()}")

plot_fit('Trained model test')


# %%

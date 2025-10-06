#%%
# Headers 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#%%
# Creating a dataset
# randn returns a tensor filled with random numbers distributed
# 100 points with normal distriubution of 1
X = torch.randn(100, 1)
# Adding noise by distributing the Y coordinates
y = X + 3 * torch.randn(100, 1)
plt.plot(X.numpy(), y.numpy(), 'o')
plt.xlabel("x")
plt.ylabel("y")


#%%
# A linear model template
# Inheriting from Module class
class LR(nn.Module):
    def __init__(self, input_size, output_size):
        # Calling parent class constructor
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        pred = self.linear(x)
        return pred


# %%
# Creating a new linear model
torch.manual_seed(1)
model = LR(1,1)
print(model)

#%%
# Getting model parameters
[w, b] = model.parameters()
# Fetching model parameters
# Using a func
def get_params():
    return (w[0][0].item(), b[0].item())

#%%
# Plotting the params
def plt_fit(title):
    plt.title = title
    w1, b1 = get_params()
    x1 = np.array([-30, 30])
    y1 = w1*x1 + b1
    plt.plot(x1, y1, 'r')
    plt.scatter(X, y)
    plt.show()

plt_fit('Initial Model')

#%%

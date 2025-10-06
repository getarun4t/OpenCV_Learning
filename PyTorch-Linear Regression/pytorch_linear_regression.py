# %%
# Defining the headers
import torch
from torch.nn import Linear

# %%
# Defining variables 
# Optimization alg used is called Gradient decent
# y = wx +b 
# w is slope/weight
# b is bias or y intercept
# using linear model we can predict y for each value of x
# requires_grad true as it requires gradient
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

#%%
# Defining forward function
# Receives input value which is x
def forward(x):
    y = w*x + b
    return y

# %%
# Testing with random input
x = torch.tensor(2)
y_predicted = forward(x)
print(y_predicted)

# %%
# Generating seeds for random values
torch.manual_seed(1)

# Creating a linear model
model = Linear(in_features=1, out_features=1)
# Print model parameters
print(model.bias, model.weight)

#%%
# Testing the model
x = torch.tensor([2.0])
y = model(x)
print(y)

# Checking with multiple inputs
x = torch.tensor([[2.0], [3.3]])
y = model(x)
print(y)
# %%

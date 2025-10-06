#%%
# Headers 
import torch
import torch.nn as nn

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

#%%
# Using forward class
x = torch.tensor([1.0])
print(model.forward(x))

# Checking with multiple inputs
x = torch.tensor([[1.0], [2.0]])
print(model.forward(x))

#%%
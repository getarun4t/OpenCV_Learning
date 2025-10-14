#%%
# Headers
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

#%%
# Getting MNIST data set
# Converting to tensor using transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 1), (0.5, 0.5, 0.5))
                                ])
training_dataset = datasets.MNIST(root = './data', train = True, download= True, transform=transform)
print(training_dataset)

#%%
#%%
# Headers
import torch 
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as py

from torchvision import transforms, models
from PIL import Image

#%%
# Importing pretrained model 
# VGG 19 model used here
# 19 layers
vgg = models.vgg19(pretrained=True).features

for param in vgg.parameters():
    # Ensures that parameters are not updated by back-propogation
    param.requires_grad_(False)

#%%
# Using GPU
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# Outputing the architecture of the model
# Input takes 3 channel coloured image
vgg.to(device)

# %%
# Load Images
def load_images(img_path, max_size=400, shape=None):
    # Opening Transforming the image to be compatible
    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    # Composing all the transforms required
    in_transform = transforms.Compose([
        # Decreses overall size of image
        # When smaller edge to 400, larger edge of image is scaled down due to aspect ratio
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Applying the transformations
    image = in_transform(image).unsqueeze(0)

    return image
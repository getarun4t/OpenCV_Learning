#%%
# Headers
import torch 
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms, models
from PIL import Image

images_folder = "../../PyTorch_Images/Images/"

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
# Creating Image Loader function
def load_images(img_path, max_size=400, shape=None):
    # Opening Transforming the image to be compatible
    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape
    
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

#%%
# Loading the images
content = load_images(images_folder+"City.jpg").to(device)
style = load_images(images_folder+"StarryNight.jpg", shape=content.shape[-2:]).to(device)

# %%
# Converting to numpy images for compatibility
def im_convert(tensor):
    image = tensor.clone().detach().cpu().numpy()
    # Removing single dimensional entries
    image = image.squeeze()
    # Getting 28*28*1 shape
    image = image.transpose (1, 2, 0)
    # Normalizing the image
    image = image* np.array((0.5, 0.5, 0.5))+ np.array((0.5, 0.5, 0.5))
    # Clipping the image
    image = image.clip(0,1)
    return image

# %%
# Plotting the images
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
ax1.imshow(im_convert(content))
ax1.axis('off')
ax2.imshow(im_convert(style))
ax2.axis('off')

# %%
# Function for getting image features
def get_features(image, model):
    # defining layers for feature extraction as a dict object
    layers = {'0': 'conv1_1',   # all others style extraction
              '5': 'conv2_1',
              '10': 'conv3_1',
              '19': 'conv4_1',
              '21': 'conv4_2',  # content extraction
              '28': 'conv5_1',}
    # Empty dict for saving features
    features={}

    for name, layer in model._modules.items():
        # Output of first layer becomes input for next
        image = layer(image)
        if name in layers:
            # store output in features dict
            features[layers[name]] = image
    
    return features

# %%
# Getting the features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

#%%
# Function to apply Gram Matrix
def gram_matrix(tensor):
    # Reshaping the tensor from 4d
    _, d, h, w = tensor.size()
    # Reshaping to 2d tensor
    # d is feature depth
    tensor = tensor.view(d, h*w)
    # Getting gram matrix
    gram = torch.mm(tensor, tensor.t())
    return gram

#%%
# Applying gram matrix to style features
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# %%
# Upper layers should have more weights for better style transfer
style_weights = {'conv1_1': 1,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,  
                 'conv5_1': 0.2}
content_weight = 1
# Can be changed and played around to find optimal weight
style_weight = 1e6 

#%%
# Getting target image
target = content.clone().requires_grad_(True).to(device)

# %%
# Basic parameters for visualizing training process
# Updated image every 300 iterations
show_every = 300
optimizer = optim.Adam([target], lr = 0.003)
# More step, lower loss value, but takes longer, min 21k steps
steps = 21000
# Shape of target array
height, width, channels = im_convert(target).shape
image_array = np.empty(shape=(300, height, width, channels))
# capturing frame every steps/300 
capture_frame = steps/300
# initial_value
counter = 0

#%%
# Optimization process
for ii in range(1, steps+1):
    # Collection of features for parent target image
    target_features = get_features(target, vgg)
    # Calculating the content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    # Calculating Style loss
    # Combined wtd avg of 5 layers
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        # Weighting each layer loss
        layer_style_loss = style_weights[layer] * torch.mean((target_gram -style_gram)**2)
        _, d, h, w = target_feature.shape 
        style_loss += layer_style_loss / (d*h*w)
    
    # Total loss of content and style loss
    # Finding weighted total loss
    total_loss = content_loss*content_weight + style_loss*style_weight

    # Resetting optimizer
    optimizer.zero_grad()
    # Configuring min loss
    total_loss.backward()
    optimizer.step()

    # Steps for data visualization
    if ii % show_every == 0:
        print('Total loss: ', total_loss.item())
        print('Iteration: ', ii)
        plt.imshow(im_convert(target))
        plt.axis('off')
        plt.show()
    
    # For a video
    if ii% capture_frame==0:
        image_array[counter] = im_convert(target)
        counter +=1

#%%
# Plotting the loss
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax1.axis('off')
ax1.imshow(im_convert(style))
ax1.axis('off')
ax1.imshow(im_convert(target))
ax1.axis('off')
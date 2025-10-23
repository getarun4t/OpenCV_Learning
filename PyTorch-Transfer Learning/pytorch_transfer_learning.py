#%%
# Headers
import torch
import requests
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import PIL
from torch import nn
from torchvision import datasets, transforms, models
from PIL import Image
from torchvision.models import AlexNet_Weights, VGG16_Weights

#%%
# Adding Cuda
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#%%
# Data augmentation
# Getting MNIST data set
# Converting to tensor using transform
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Normalize((0.5,), (0.5,))
])

# Getting MNIST data set
# Converting to tensor using transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# Dataset - https://github.com/jaddoescad/ants-bees-dataset
# Creating training and validation dataset
training_dataset = datasets.ImageFolder(root = '../../ants-bees-dataset/train',  transform=transform_train)
validation_dataset = datasets.ImageFolder(root = '../../ants-bees-dataset/val',  transform=transform)
print(len(training_dataset))
print(len(validation_dataset))
# Cutting to smaller chunks
training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=100, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=100, shuffle=False)

#%%
# Changing to numpy array
def im_convert(tensor):
    image = tensor.clone().detach().numpy()
    # Getting 28*28*1 shape
    image = image.transpose (1, 2, 0)
    # Normalizing the image
    image = image* np.array((0.5, 0.5, 0.5))+ np.array((0.5, 0.5, 0.5))
    # Clipping the image
    image = image.clip(0,1)
    return image

#%%
classes = ('ants', 'bees')

#%%
# Creating an iterable
data_iter = iter(training_loader)
images, labels = next(data_iter)
fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx+1)
    plt.imshow(im_convert(images[idx]))
    ax.set_title(classes[labels[idx].item()])

#%%
# Loading AlexNet Model
model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
print(model)

#%%
# Adjusting model
for param in model.features.parameters():
    # When gradients are not used 
    param.requires_grad = False

#%%
# Update layer layer in the model to output 2 classes only
# Selecting inputs to 6th layer in model and modifying
n_inputs = model.classifier[6].in_features
# Creating a new last year with 2 classes as output
# This will be the final layer
last_year = nn.Linear(n_inputs, len(classes))
# Replacing last layer with anove
model.classifier[6] = last_year
print(model)

#%%
# Moving model to GPU
model.to(device)

#%%
# Getting the loss function
criterion = nn.CrossEntropyLoss()
# Increasing training rate 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

#%%
# Training 
epochs = 5
running_loss_history = []
running_correct_history = []
validation_loss_history = []
validation_correct_history = []

for e in range(epochs):
    running_loss = 0.0
    running_correct = 0.0

    validation_running_loss = 0.0
    validation_running_correct = 0.0

    for images, labels in training_loader:
        # Using GPU
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)

        running_loss+=loss.item()
        running_correct+=torch.sum(preds == labels.data)
    else:
        with torch.no_grad():
            for val_inputs, val_labels in validation_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                _, val_preds = torch.max(val_outputs, 1)

                validation_running_loss+=val_loss.item()
                validation_running_correct+=torch.sum(val_preds == val_labels.data)
        epoch_loss = running_loss/len(training_loader.dataset)
        epoch_acc = running_correct.float()/len(training_loader.dataset)
        running_loss_history.append(epoch_loss)
        running_correct_history.append(epoch_acc)
        
        val_epoch_loss = validation_running_loss/len(validation_loader.dataset)
        val_epoch_acc = validation_running_correct.float()/len(validation_loader.dataset)
        validation_loss_history.append(val_epoch_loss)
        validation_correct_history.append(val_epoch_acc)

        print(f'epoch: {e+1}')
        print(f'Training loss : {epoch_loss}, Training Accuracy: {epoch_acc.item()}')
        print (f'Validation loss: {val_epoch_loss}, Validation Accuracy: {val_epoch_acc.item()}')

#%%
# Converting to CPU data for plotting
running_correct_history = [x.cpu().item() for x in running_correct_history]
validation_correct_history = [x.cpu().item() for x in validation_correct_history]

#%%
# Plotting loss
plt.plot(running_loss_history, label='training loss')
plt.plot(validation_loss_history, label='Validation loss')
plt.legend()

#%%
# Plotting accuracy
plt.plot(running_correct_history, label='Accuracy')
plt.plot(validation_correct_history, label='Validation Accuracy')
plt.legend()

# %%
# Getting test image from web
url = 'https://e3.365dm.com/23/09/2048x1152/skynews-red-fire-ant-generic_6282149.jpg'
response = requests.get(url, stream=True)
img = Image.open(response.raw).convert('RGB')
plt.imshow(img)

#%%
# Preprocessing the image to model input format 
# Change image to black background and white digit
# Transforming to 32*32
img = img.resize((32, 32)) 

# Transform through same MNIST preprocessing 
img = transform(img)
plt.imshow(im_convert(img))

#%%
# Adding to the model, flattening and predicting
img = img.to(device)

if len(img.shape) == 3:  # if missing batch dimension, add one
    img = img.unsqueeze(0)

# DO NOT FLATTEN BEFORE PASSING TO CNN
output = model(img)  
_, pred = torch.max(output, 1)
print(classes[pred.item()])

# %%
# Validation iter
# Creating an iterable
data_iter = iter(validation_loader)
images, labels = next(data_iter)

# Using GPU
images = images.to(device)
labels = labels.to(device)

# ✅ Make sure shape is correct: [B, C, H, W]
# For MNIST-like data, it should already be like this: [batch, 1, 28, 28]
print(images.shape)  # Expected: torch.Size([100, 1, 28, 28])

# ✅ Run forward pass without flattening
output = model(images)

_, preds = torch.max(output, 1)

# Plot
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx+1)
    plt.imshow(im_convert(images[idx].cpu()))
    ax.set_title(f"{str(classes[preds[idx].item()])} ({str(classes[labels[idx].item()])})",
                 color="green" if preds[idx] == labels[idx] else "red")
# %%

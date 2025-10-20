#%%
# Headers
import torch
import requests
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import PIL
from torch import nn
from torchvision import datasets, transforms
from PIL import Image

#%%
# Getting MNIST data set
# Converting to tensor using transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# Creating training and validation dataset
training_dataset = datasets.MNIST(root = './data', train = True, download= True, transform=transform)
validation_dataset = datasets.MNIST(root = './data', train = False, download= True, transform=transform)
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
# Creating an iterable
data_iter = iter(training_loader)
images, labels = next(data_iter)
fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx+1)
    plt.imshow(im_convert(images[idx]))
    ax.set_title([labels[idx].item()])

#%%
# Using LeNet Model
class LeNet(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super().__init__()
        # First convolutional layer (Input layer)
        # 1 input layer as greyscale, 20 output layer, kernel scale 5, strive length 1 as input is small  
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # Second layer
        # 50 output layer as output
        self.conv1 = nn.Conv2d(20, 50, 5, 1)
        # Fully connected layers
        # Padding can be added to prevent size reduction (not used now)
        # 50 channels input with 4*4
        self.fc1 = nn.Linear(4*4*50, 500)
        # Second fc layer
        # Output is 10 as MNIST has 10 classes to be classified
        self.fc1 = nn.Linear(500, 10)
    
    def forward(self, x):
        # First pooling layer
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        # Second pooling layer
        x = F.relu(self.conv2(x))
        # Pooling layer cuts image size by 2
        x = F.max_pool2d(x, 2, 2)
        # After final pooling layer, image has to be flattened before going to fully connected layer
        x = x.view(-1, 4*4*50)
        # Attaching relu activation function to fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Setting hidden layer dimensions during init    
model = Classifier(784,125, 65, 10)
print(model)

#%%
# Getting the loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

#%%
# Setting the number of epochs
epochs = 12
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
        # Reshaping the training 
        inputs = images.view(images.shape[0], -1)
        outputs = model(inputs)
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
                val_inputs = val_inputs.view(val_inputs.shape[0], -1)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                _, val_preds = torch.max(val_outputs, 1)

                validation_running_loss+=val_loss.item()
                validation_running_correct+=torch.sum(val_preds == val_labels.data)
        epoch_loss = running_loss/len(training_loader)
        epoch_acc = running_correct.float()/len(training_loader)
        running_loss_history.append(epoch_loss)
        running_correct_history.append(epoch_acc)
        
        val_epoch_loss = validation_running_loss/len(validation_loader)
        val_epoch_acc = validation_running_correct.float()/len(validation_loader)
        validation_loss_history.append(val_epoch_loss)
        validation_correct_history.append(val_epoch_acc)

        print(f'epoch: {e+1}')
        print(f'Training loss : {epoch_loss}, Training Accuracy: {epoch_acc.item()}')
        print (f'Validation loss: {val_epoch_loss}, Validation Accuracy: {val_epoch_acc.item()}')

#%%
# Plotting loss
plt.plot(running_loss_history, label='training loss')
plt.plot(validation_loss_history, label='Validation loss')

#%%
# Plotting accuracy
plt.plot(running_correct_history, label='Accuracy')
plt.plot(validation_correct_history, label='Validation Accuracy')

# %%
# Getting test image from web
url = 'https://images.homedepot-static.com/productImages/007164ea-d47e-4f66-8d8c-fd9f621984a2/svn/architectural-mailboxes-house-letters-numbers-3585b-5-64_1000.jpg'
response = requests.get(url, stream=True)
img = Image.open(response.raw)
plt.imshow(img)

#%%
# Preprocessing the image to model input format 
# Change image to black background and white digit
img = PIL.ImageOps.invert(img)
# Converting the image to single channel
img = img.convert('L')
# Transforming to 28*28
img = img.resize((28, 28)) 

# Transform through same MNIST preprocessing 
img = transform(img)
plt.imshow(im_convert(img))

#%%
# Adding to the model, flattening and predicting
img = img.view(img.shape[0], -1)
output = model(img)
_, pred = torch.max(output, 1)
print(pred.item())

# %%
# Validation iter
# Creating an iterable
data_iter = iter(validation_loader)
images, labels = next(data_iter)

# ✅ Keep original for display
images_display = images.clone()

# ✅ Flatten ONLY for prediction
inputs = images.view(images.shape[0], -1)
output = model(inputs)
_, preds = torch.max(output, 1)

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx+1)
    plt.imshow(im_convert(images_display[idx]))
    correct = preds[idx] == labels[idx]
    ax.set_title(f"{preds[idx].item()} ({labels[idx].item()})", color="green" if correct else "red")
    ax.axis("off")
# %%

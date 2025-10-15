#%%
# Headers
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

#%%
# Getting MNIST data set
# Converting to tensor using transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
training_dataset = datasets.MNIST(root = './data', train = True, download= True, transform=transform)
# Cutting to smaller chunks
training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=100, shuffle=True)

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
# Creating a neural network class
class Classifier(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super().__init__()
        # Input layer
        self.linear1 = nn.Linear(D_in, H1)
        # First hidden layer
        self.linear2 = nn.Linear(H1, H2)
        # Second hidden layer
        self.linear3 = nn.Linear(H2, D_out)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # No activation required in last layer
        # Output returned is raw output
        x = self.linear3(x)
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

for e in range(epochs):
    running_loss = 0.0
    running_correct = 0.0

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
        epoch_loss = running_loss/len(training_loader)
        epoch_acc = running_correct.float()/len(training_loader)
        running_loss_history.append(epoch_loss)
        print(f'Training loss: {epoch_loss}')
        print(f'Accuracy: {epoch_acc}')

#%%
# Plotting 
plt.plot(running_loss_history, label='training loss')

#%%
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
images, labels = data_iter.next()
fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx+1)
    plt.imshow(im_convert(images[idx]))
    ax.set_title([labels[idx].item()])

#%%
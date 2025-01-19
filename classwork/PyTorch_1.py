"""
Learning Tensors which will be handy when working with deeper levels
like images or more complex structures.

We will aim to understand how PyTorch simplifies the processes of deep learning
"""

import torch
import matplotlib.pyplot as plt

#create 3D tensor
images = torch.rand((4,28,28))

#Get the second image
second_image = images[1]

plt.imshow(second_image, cmap ='gray') #Display
plt.axis('off') #disable axes
plt.show()
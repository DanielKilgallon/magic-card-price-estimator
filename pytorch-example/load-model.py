import numpy as np
import torchvision
import os
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

file_path = '../images/'
files = os.listdir(file_path)
image_arr = []

for file_name in files:
    image = Image.open(file_path + file_name)
    print(image)
    # show the image
    # image.show()

    # transform Image into the numpy array
    image_2_npArray = np.asarray(image)

    # transform the numpy array into the tensor
    image_2_npArray_2_tensor = torchvision.transforms.ToTensor()(image_2_npArray)
    image_arr.append(image_2_npArray_2_tensor)
    break

    #### Stage 2: transfer torch.Tensor back to Image
    npArray_2_image = torchvision.transforms.ToPILImage(mode=None)(image_2_npArray_2_tensor)
    print(npArray_2_image)
    # npArray_2_image.save('2432_recovery.jpg')
    Image._show(npArray_2_image)
    break

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_prelu_stack = nn.Sequential(
            nn.Linear(25*25, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.PReLU(),
            nn.Linear(512, 10),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_prelu_stack(x)
        return logits
model = NeuralNetwork().to(device)
image_arr = np.array(image_arr)
logits = model(image_arr)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
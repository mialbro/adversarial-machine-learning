import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, 
import matplotlib.pyplot as plt

import torch.optim as optim

from PIL import Image

class GPDA(nn.Module):
    def __init__(self, model):
        super(GPDA, self).__init__()

    def forwared(self, image):
        x = model(image)
        return x
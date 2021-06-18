import torch 
import torch.nn as nn


class GPDA(nn.Module):
    def __init__(self, model, Pt)
        super(GPDA, self).__init__()
        self.model = model
        self.Pt = Pt

        self.relu = torch.nn.ReLU()
        self.conv1 = nn.Conv2d(16, 22, 3)
        self.conv2 = nn.Conv2d(22, 32, 3)
        self.conv3 = nn.Conv2d(32, 45, 3)
        self.conv4 = nn.Conv2d(45, 64, 3)
        self.conv5 = nn.Conv2d(64, 22, 3)
        self.conv6 = nn.Conv2d(16, 22, 3)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(22)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(45)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(3)

    def forward(self, image):
        B, N, D = image.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv12(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        return x
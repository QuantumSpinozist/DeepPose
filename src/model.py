
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepPoseRegressor(nn.Module):
    def __init__(self):
        super(DeepPoseRegressor, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)  # C(55 x 55 x 96)
        self.lrn1 = nn.LocalResponseNorm(size=5)  # LRN
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  # P

        # Second Convolutional Block
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)  # C(27 x 27 x 256)
        self.lrn2 = nn.LocalResponseNorm(size=5)  # LRN
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)  # P

        # Third Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)  # C(13 x 13 x 384)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)  # C(13 x 13 x 384)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)  # C(13 x 13 x 256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)  # P

        # Fully Connected Layers
        self.fc1 = nn.Linear(6400, 4096)  # F(4096)
        self.fc2 = nn.Linear(4096, 28)  # F(4096)
        
    def forward(self, x):
        # Forward pass through the network
        x = self.conv1(x)
        x = F.relu(x)
        x = self.lrn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.lrn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        return x.view(-1, 14, 2)
    
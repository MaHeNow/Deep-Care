import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class ConvNetv1(nn.Module):

    def __init__(self):
        super(ConvNetv1, self).__init__()
        # 4 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(4, 64, 1)
        self.conv2 = nn.Conv2d(64, 128, 2)
        self.conv3 = nn.Conv2d(128, 256, 3)
        #self.conv4 = nn.Conv2d(256, 256, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(256* 11 * 5, 10000)  # 6*6 from image dimension
        self.fc2 = nn.Linear(10000, 5000)
        self.fc3 = nn.Linear(5000, 800)
        self.fc4 = nn.Linear(800, 200)
        self.fc5 = nn.Linear(200, 50)
        self.fc6 = nn.Linear(50, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
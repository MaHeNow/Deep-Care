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


class ConvNetW51H100V1(nn.Module):

    def __init__(self):
        super(ConvNetW51H100V1, self).__init__()
        # 4 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(4, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 23 * 11, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def conv_net_w51_h100_v1():
    return ConvNetW51H100V1()


class ConvNetW51H100V2(nn.Module):

    def __init__(self):
        super(ConvNetW51H100V2, self).__init__()
        # 4 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(4, 6, (5,3))
        self.conv2 = nn.Conv2d(6, 16, (7,5))
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 21 * 10, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def conv_net_w51_h100_v2():
    return ConvNetW51H100V2()


class ConvNetW51H100V3(nn.Module):

    def __init__(self):
        super(ConvNetW51H100V3, self).__init__()
        # 4 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(4, 6, (7,3))
        self.conv2 = nn.Conv2d(6, 16, (9,5))
        self.conv3 = nn.Conv2d(16, 16, (11,7))
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 2, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def conv_net_w51_h100_v3():
    return ConvNetW51H100V3()


class ConvNetW51H100V4(nn.Module):
    # 96% accuracy on unseen data

    def __init__(self):
        super(ConvNetW51H100V4, self).__init__()
        # 4 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(4, 6, (7,3))
        self.conv2 = nn.Conv2d(6, 16, (9,5))
        self.conv3 = nn.Conv2d(16, 16, (11,7))
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 19 * 10, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def conv_net_w51_h100_v4():
    return ConvNetW51H100V4()


class ConvNetW51H100V5(nn.Module):

    def __init__(self):
        super(ConvNetW51H100V5, self).__init__()

        # Convolutional part of the network with batch-normalization inbetween
        # the convolutional layers
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=6, kernel_size=(7,3)),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),  nn.BatchNorm2d(6),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(9,5)),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),  nn.BatchNorm2d(16),
        )

        # Fully connected dense part of the network with dropout inbetween
        # the linear layers
        self.dense = nn.Sequential(
            nn.Linear(16 * 19 * 10, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(84, 4)
        )


    def forward(self, x):
        # Apply convolutional network
        x = self.convolution(x)

        # Flatten the output of the convolutional layer for the linear layer
        x = x.view(-1, 16 * 19 * 10)
 
        # Apply the fully connected dense layer and classify the image
        x = self.dense(x)
        return x


def conv_net_w51_h100_v5():
    return ConvNetW51H100V5()


class ConvNetW51H100V6(nn.Module):

    def __init__(self):
        super(ConvNetW51H100V6, self).__init__()

        # Convolutional part of the network with batch-normalization inbetween
        # the convolutional layers
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=10, kernel_size=(3,3)),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),  nn.BatchNorm2d(10),
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(4,4)),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),  nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5)),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),  nn.BatchNorm2d(64),
        )

        # Fully connected dense part of the network with dropout inbetween
        # the linear layers
        self.dense = nn.Sequential(
            nn.Linear(64 * 9 * 3, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(200, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(84, 4)
        )


    def forward(self, x):
        # Apply convolutional network
        x = self.convolution(x)

        # Flatten the output of the convolutional layer for the linear layer
        x = x.view(-1, 64 * 9 * 3)
 
        # Apply the fully connected dense layer and classify the image
        x = self.dense(x)
        return x


def conv_net_w51_h100_v6():
    return ConvNetW51H100V6()


class ConvNetW51H100V7(nn.Module):

    def __init__(self):
        super(ConvNetW51H100V7, self).__init__()

        # Convolutional part of the network with batch-normalization inbetween
        # the convolutional layers
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=10, kernel_size=(3,3)),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),  nn.BatchNorm2d(10),
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(5,5)),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),  nn.BatchNorm2d(16),
        )

        # Fully connected dense part of the network with dropout inbetween
        # the linear layers
        self.dense = nn.Sequential(
            nn.Linear( 16 * 22 * 10, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(120, 4)
        )


    def forward(self, x):
        # Apply convolutional network
        x = self.convolution(x)

        # Flatten the output of the convolutional layer for the linear layer
        x = x.view(-1, 16 * 22 * 10)
 
        # Apply the fully connected dense layer and classify the image
        x = self.dense(x)
        return x


def conv_net_w51_h100_v7():
    return ConvNetW51H100V7()


class ConvNetW51H100V8(nn.Module):

    def __init__(self):
        super(ConvNetW51H100V8, self).__init__()

        # Convolutional part of the network with batch-normalization inbetween
        # the convolutional layers
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=10, kernel_size=(3,3)),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),  #nn.BatchNorm2d(10),
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(5,5)),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),  #nn.BatchNorm2d(16),
        )

        # Fully connected dense part of the network with dropout inbetween
        # the linear layers
        self.dense = nn.Sequential(
            nn.Linear( 16 * 22 * 10, 120),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.3),
            nn.Linear(120, 4)
        )


    def forward(self, x):
        # Apply convolutional network
        x = self.convolution(x)

        # Flatten the output of the convolutional layer for the linear layer
        x = x.view(-1, 16 * 22 * 10)
 
        # Apply the fully connected dense layer and classify the image
        x = self.dense(x)
        return x


def conv_net_w51_h100_v8():
    return ConvNetW51H100V8()




class ConvNetW250H50V1(nn.Module):
    # 96% Accuracy on unseen data

    def __init__(self):
        super(ConvNetW250H50V1, self).__init__()
        # 4 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(4, 6, (7,3))
        self.conv2 = nn.Conv2d(6, 16, (9,5))
        #self.conv3 = nn.Conv2d(16, 16, (11,5))
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 7 * 60, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        #x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def conv_net_w250_h50_v1():
    return ConvNetW250H50V1()


class ConvNetW250H50V2(nn.Module):
    # 92% Accuracy on unseen data

    def __init__(self):
        super(ConvNetW250H50V2, self).__init__()
        # 4 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(4, 6, (4,4))
        self.conv2 = nn.Conv2d(6, 16, (6,5))
        self.conv3 = nn.Conv2d(16, 16, (7,6))
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 1 * 27, 160)  # 6*6 from image dimension
        self.fc2 = nn.Linear(160, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def conv_net_w250_h50_v2():
    return ConvNetW250H50V2()


class ConvNetW250H50V3(nn.Module):
    # Accuracy von 96.88 auf unseen data

    def __init__(self):
        super(ConvNetW250H50V3, self).__init__()
        # 4 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(4, 6, (4,4))
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        #self.conv3 = nn.Conv2d(16, 16, (7,6))
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 9 * 59, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        #x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def conv_net_w250_h50_v3():
    return ConvNetW250H50V3()
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNetW51H101V1(nn.Module):

    def __init__(self):
        super(ConvNetW51H101V1, self).__init__()

        # Convolutional part of the network with batch-normalization inbetween
        # the convolutional layers
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=24, kernel_size=(3,3)),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True), # nn.BatchNorm2d(24),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(4,4)),
            nn.MaxPool2d(3, 3), nn.ReLU(inplace=True), # nn.BatchNorm2d(48),
        )

        # Fully connected dense part of the network with dropout inbetween
        # the linear layers
        self.dense = nn.Sequential(
            nn.Linear(48 * 31 * 35, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(500, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            #nn.Linear(200, 32),
            #nn.ReLU(inplace=True),
            #nn.Dropout(p=0.5),
            #nn.Linear(84, 32),
            #nn.ReLU(inplace=True),
            #nn.Dropout(p=0.3),
            nn.Linear(32, 4)
        )


    def forward(self, x):
        # Apply convolutional network
        x = self.convolution(x)

        # Flatten the output of the convolutional layer for the linear layer
        x = x.view(-1, 48 * 31 * 35)
 
        # Apply the fully connected dense layer and classify the image
        x = self.dense(x)
        return x


def conv_net_w221_h221_v6():
    return ConvNetW51H101V1()

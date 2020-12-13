import os
import glob
import numpy as np
import torch
from deepcare.utils import msa

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 4 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(4, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 11 * 4, 120)  # 6*6 from image dimension
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

msa_data_path = "/home/mnowak/care/care-output/"
msa_file_name = "humanchr1430covMSA_"

fastq_file_path = "/share/errorcorrection/datasets/artmiseqv3humanchr14" # /humanchr1430cov_errFree.fq"
fastq_file_name = "humanchr1430cov_errFree.fq.gz"
fastq_file = os.path.join(fastq_file_path, fastq_file_name)


if __name__ == "__main__":
    
    msas, labels = msa.make_examples_from_file(
        msa_file=os.path.join(msa_data_path, msa_file_name+"1"),
        reference_fastq_file=fastq_file,
        image_height=50,
        image_width=25,
        number_examples=3
    )
    print("Done creating the generator...")

    net = Net()
    print(net)

    target = torch.randn(4)  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    
    for msa in msas:
        output = net(msa.unsqueeze_(0))
        print(output)
        loss = criterion(output, target)
        print(loss)
        break

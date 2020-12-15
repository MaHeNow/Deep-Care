import os
import glob
import numpy as np
import torch
from deepcare.utils import msa

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Avalable device: ", device)
    
    # Hyperparameters
    in_channel = 3
    num_classes = 2
    learning_rate = 1e-3
    batch_size = 10
    num_epochs = 10

    # Load the data
    dataset = msa.MSADataset(
        msa_file_path=os.path.join(msa_data_path, msa_file_name+"1"),
        ref_fastq_path=fastq_file,
        image_height=50,
        image_width=25,
        number_examples=1000
    )
    print("Done generating the Dataset")

    # Create dataloaders
    train_set, test_set = torch.utils.data.random_split(dataset, [800, 200])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    # Model
    net = Net()
    print(net)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    # Train Network
    for epoch in range(num_epochs):

        running_loss = 0.0
        for i, (data, targets) in enumerate(train_loader):

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(data)
            print("outputs: ", outputs)
            print("targets: ", targets)
            loss = criterion(outputs, targets)
            print("loss: ", loss)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if  i % 10 == 9:
                print("outputs: ", outputs)
                print("targets: ", targets)
                print("loss: ", loss)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
         
    print('Finished Training')

    # relic:
    #for msa, label in zip(msas, labels):
    #    output = net(msa.unsqueeze_(0))
    #    print(output)
    #    loss = criterion(output, label.unsqueeze_(0))
    #    print(loss)
    #    break

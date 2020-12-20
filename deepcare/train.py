from dataset import MSADataset
from models.conv_net import ConvNetv1

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 4 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(4, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 23 * 1, 120)  # 6*6 from image dimension
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


# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
    
    model.train()


if __name__ == "__main__":

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 50 
    learning_rate = 0.00001
    train_CNN = False
    batch_size = 128
    shuffle = True
    pin_memory = True
    num_workers = 1
    
    transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
            ]
        )

    dataset = MSADataset(
        root_dir="datasets/humanchr1430covMSA_center_base_dataset_w51_h100_n64000_human_readable",
        annotation_file="datasets/humanchr1430covMSA_center_base_dataset_w51_h100_n64000_human_readable/train_labels.csv",
        transform=transform
        )

    train_set, validation_set = torch.utils.data.random_split(dataset,[52000,12000])
    train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
    validation_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)

    # Model
    model = ConvNetv1()
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    print("Starting the training process.")

    # Train Network
    for epoch in range(num_epochs):
        losses = []
        
        epoch_start_time = time.time()
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)
            
            # forward
            scores = model(data)
            loss = criterion(scores, targets)
            
            losses.append(loss.item())
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            
            # gradient descent or adam step
            optimizer.step()
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Cost at epoch {epoch} is {sum(losses)/len(losses)}. Training the epoch took {epoch_duration} seconds.')

        if epoch+1 % 10 == 0:
            print(f"Accuracy on the validation set is {check_accuracy(validation_loader, model)}")
    
    end_time = time.time()
    duration = end_time - start_time

    print(f"Finished training. The process took {duration} seconds.")    
    
    print("Checking accuracy on Training Set")
    check_accuracy(train_loader, model)

    print("Checking accuracy on Test Set")
    check_accuracy(validation_loader, model)

    save_path = "trained_models"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(model.state_dict(), os.path.join(save_path, "conv_net_humanchr1430_center_base_w51_h100_n64000"))

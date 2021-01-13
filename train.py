import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from deepcare.data import MSADataset
from deepcare.models.conv_net import conv_net_w51_h100_v1, conv_net_w51_h100_v3, conv_net_w51_h100_v4, conv_net_w250_h50_v1, conv_net_w250_h50_v3
from deepcare.utils.accuracy import check_accuracy, check_accuracy_on_classes

if __name__ == "__main__":

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 180
    learning_rate = 0.00001
    batch_size = 256
    shuffle = True
    pin_memory = True
    num_workers = 20
    dataset_folder = "datasets"
    dataset_name = "humanchr1430covMSA_non_hq_part4_center_base_dataset_w51_h100_n279295_not_human_readable"
    #validationset_name = "humanchr1430covMSA_part1_center_base_dataset_w51_h100_n24000_human_readable"
    dataset_csv_file = "train_labels.csv"
    model_out_dir = "trained_models"
    model_name = "conv_net_v4_humanchr1430_center_base_w51_h100_n384000_not_human_readable_part4_non_hq"

    # Model
    model = conv_net_w51_h100_v4()
    model.to(device)

    transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
            ]
        )

    train_set = MSADataset(
        root_dir=os.path.join(dataset_folder, dataset_name),
        annotation_file=os.path.join(dataset_folder, dataset_name, dataset_csv_file),
        transform=transform
        )
    
    #validation_set = MSADataset(
    #    root_dir=os.path.join(dataset_folder, validationset_name),
    #    annotation_file=os.path.join(dataset_folder, validationset_name, dataset_csv_file),
    #    transform=transform
    #)

    train_set, validation_set = torch.utils.data.random_split(train_set,[209472, 69824])
    train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
    validation_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)

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

        if (epoch+1) % 20 == 0:
            print(f"Accuracy on the validation set after epoch {epoch}:") 
            check_accuracy(validation_loader, model, device)

    end_time = time.time()
    duration = end_time - start_time

    print(f"Finished training. The process took {duration} seconds.")    

    print("Checking accuracy on Training Set")
    check_accuracy_on_classes(train_loader, model, device)

    print("Checking accuracy on Test Set")
    check_accuracy_on_classes(validation_loader, model, device)

    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)

    torch.save(model.state_dict(), os.path.join(model_out_dir, model_name))

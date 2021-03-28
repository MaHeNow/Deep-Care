import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from deepcare.data import MSADataset
from deepcare.models.conv_net import *

from deepcare.utils.accuracy import check_accuracy, check_accuracy_on_classes


if __name__ == "__main__":

    # Check if training on the GPU is possible
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    #torch.cuda.empty_cache()


    # -------------- Hyperparameters -------------------------------------------
    num_epochs = 15
    learning_rate = 0.00001
    batch_size = 1024
    shuffle = True
    pin_memory = True
    num_workers = 60


    # -------------- File structure --------------------------------------------
    dataset_folder = "/home/mnowak/data/quality_datasets/w221_h221"
    dataset_name = "artmiseqv3humanchr1430covMSATraining/examples"
    validationset_name = "artmiseqv3humanchr1430covMSAValidation/examples"
    dataset_csv_file = "artmiseqv3humanchr1430covMSATraining/train_labels.csv"
    validationset_csv_file = "artmiseqv3humanchr1430covMSAValidation/train_labels.csv"
    existing_model_path = ""

    model_out_dir = "/home/mnowak/data/trained_models/conv_net_w221_h221_v6/BalancedHmnChr14DSetRetrained3/"
    model_name = "state_dict"

    # Create the output directory if it does not exist yet
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)

    # -------------- Preparing the Model ---------------------------------------
    model = conv_net_w221_h221_v6()
    if existing_model_path != "":
        state_dict = torch.load(existing_model_path)
        model.load_state_dict(state_dict)
    model.to(device)


    # -------------- Defining the Loss-Function and Optimizer ------------------ 
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # -------------- Evaluation Dataframes -------------------------------------
    meta_df_data = {
        "Category" : ["Epochs", "Optimizer", "Criteroin", "Batch Size", "Learning Rate", "Shuffle", "Pin Memory", "Number Workers", "Dataset Path"],
        "Value" : [num_epochs, str(optimizer).replace("\n", ""), str(criterion), batch_size, learning_rate, shuffle, pin_memory, num_workers, os.path.join(dataset_folder, dataset_name)]
        }
    columns=["epoch", "training_loss", "validation_loss", "training_time"]
    training_df_data = {column : [] for column in columns}


    # -------------- Trandformation for the Datasets ---------------------------
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
            ])


    # -------------- Getting the Training and Validation Datasets --------------
    train_set = MSADataset(
        root_dir=os.path.join(dataset_folder, dataset_name),
        annotation_file=os.path.join(dataset_folder, dataset_csv_file),
        transform=transform
        )
    validation_set = MSADataset(
        root_dir=os.path.join(dataset_folder, validationset_name),
        annotation_file=os.path.join(dataset_folder, validationset_csv_file),
        transform=transform
        )
    meta_df_data["Category"].append("Number Training Samples")
    meta_df_data["Value"].append(train_set.__len__())
    meta_df_data["Category"].append("Number Validation Samples")
    meta_df_data["Value"].append(validation_set.__len__())

    # -------------- Creating Dataloaders --------------------------------------
    train_loader = DataLoader(
        dataset=train_set, 
        shuffle=shuffle, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
        )
    validation_loader = DataLoader(
        dataset=validation_set, 
        shuffle=shuffle, 
        batch_size=batch_size,
        num_workers=num_workers, 
        pin_memory=pin_memory
        )


    # -------------- Starting the Training Process -----------------------------
    start_time = time.time()
    print("Starting the training process.")

    for epoch in range(num_epochs):
        losses = []
        val_losses = []
        
        epoch_start_time = time.time()
        for (data, targets) in tqdm(train_loader, ascii=True, desc=f"Epoch: {epoch}"):

            # get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward
            scores = model(data)
            loss = criterion(scores, targets)
            losses.append(loss.item())
            
            # backward
            loss.backward()
            
            # gradient descent or adam step
            optimizer.step()
        
        epoch_end_time = time.time()

        
        if (epoch+1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                for (val_data, val_targets) in tqdm(validation_loader, ascii=True, desc=f"Validation Epoch: {epoch}"):

                    val_data = val_data.to(device=device)
                    val_targets = val_targets.to(device=device)

                    # test the validation data for further analysis
                    val_scores = model(val_data)
                    val_loss = criterion(scores, targets)
                    val_losses.append(val_loss.item())

            model.train()
            val_loss_at_epoch = sum(val_losses)/len(val_losses)

        else:

            val_loss_at_epoch = 0
        
        epoch_duration = epoch_end_time - epoch_start_time
        loss_at_epoch = sum(losses)/len(losses)

        # Add new entries for the training dataframe
        training_df_data["epoch"].append(epoch)
        training_df_data["training_loss"].append(loss_at_epoch)
        training_df_data["validation_loss"].append(val_loss_at_epoch)
        training_df_data["training_time"].append(epoch_duration)

        print(f'Loss at epoch {epoch} is {sum(losses)/len(losses)}. Training the epoch took {epoch_duration} seconds.')
        print(f"Validation loss is {val_loss_at_epoch}.")

        # Save a checkpoint every epoch

        # Save models state dict
        torch.save(model.state_dict(), os.path.join(model_out_dir, model_name))

        # Save the csv files with the training evaluation and meta data 
        training_df = pd.DataFrame(training_df_data)
        meta_df = pd.DataFrame(meta_df_data)
        training_df.to_csv(os.path.join(model_out_dir, "training_data.csv"), index = False, header=True)
        meta_df.to_csv(os.path.join(model_out_dir, "meta_data.csv"), index = False, header=True)


    end_time = time.time()
    duration = end_time - start_time
    meta_df_data["Category"].append("Training Duration")
    meta_df_data["Value"].append(duration)

    print(f"Finished training. The process took {duration} seconds.")    

    print("Checking accuracy on Training Set")
    train_accuracy = check_accuracy(train_loader, model, device)

    print("Checking accuracy on Validation Set")
    validation_accuracy = check_accuracy(validation_loader, model, device)

    meta_df_data["Category"].append("Training Accuracy")
    meta_df_data["Value"].append(train_accuracy)
    meta_df_data["Category"].append("Validation Accuracy")
    meta_df_data["Value"].append(validation_accuracy)

    # Create the output directory if it does not exist yet
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)

    # Save models state dict
    torch.save(model.state_dict(), os.path.join(model_out_dir, model_name))

    # Save the csv files with the training evaluation and meta data 
    training_df = pd.DataFrame(training_df_data)
    meta_df = pd.DataFrame(meta_df_data)
    training_df.to_csv(os.path.join(model_out_dir, "training_data.csv"), index = False, header=True)
    meta_df.to_csv(os.path.join(model_out_dir, "meta_data.csv"), index = False, header=True)

    # Save the model for later usage with the LibTorch library
    model.to("cpu")
    model.eval()

    script_module = torch.jit.script(model)
    script_module.save(os.path.join(model_out_dir, "script_module.pt"))

from dataset import MSADataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


if __name__ == "__main__":

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 10
    learning_rate = 0.00001
    train_CNN = False
    batch_size = 32
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
        root_dir="datasets/center_base_dataset_w11_h100_n800_human_readable",
        annotation_file="center_base_dataset_11_100_human_readable/train_labels.csv",
        transform=transform
        )

    train_set, validation_set = torch.utils.data.random_split(dataset,[20000,4000])
    train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
    validation_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)

    for batch_idx, (data, targets) in enumerate(train_loader):
        print(batch_idx)
        single_msa = data[0]
        middle_row = single_msa[:,49,:]
        print(middle_row)
        print(targets)
        break

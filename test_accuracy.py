import os

from glob import glob

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from deepcare.utils.accuracy import check_accuracy_on_classes, check_accuracy
from deepcare.data import MSADataset
from deepcare.models.conv_net import *

if __name__ == "__main__":

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    dataset_folder = "/home/mnowak/data/quality_datasets/w221_h221/artmiseqv3humanchr1430covMSATraining"
    dataset_name = "examples"

    model_path = "/home/mnowak/data/trained_models/conv_net_w221_h221_v4/BalancedHmnChr14DSet/conv_net_v4_state_dict"
    model = conv_net_w221_h221_v4()

    shuffle = True
    batch_size = 256
    pin_memory = True
    num_workers = 60

    transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
            ]
        )

    dataset = MSADataset(
        root_dir=os.path.join(dataset_folder, dataset_name),
        annotation_file=os.path.join(dataset_folder, "train_labels.csv"),
        transform=transform
        )

    loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)

    check_accuracy(loader=loader, model=model, device=device)
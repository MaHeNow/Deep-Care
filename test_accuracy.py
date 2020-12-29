import os

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from deepcare.utils.accuracy import check_accuracy_on_classes
from deepcare.data import MSADataset
from deepcare.models.conv_net import conv_net_w51_h100


if __name__ == "__main__":

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = "trained_models"
    model_name = "simple_conv_net_humanchr1430_center_base_w51_h100_n64000_not_human_readable"

    shuffle = True
    batch_size = 256
    pin_memory = True
    num_workers = 1
    
    transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
            ]
        )

    dataset = MSADataset(
        root_dir="datasets/humanchr1430covMSA_center_base_dataset_w51_h100_n64000_not_human_readable",
        annotation_file="datasets/humanchr1430covMSA_center_base_dataset_w51_h100_n64000_not_human_readable/train_labels.csv",
        transform=transform
        )

    loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
    
    model = conv_net_w51_h100()
    model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
    model.to(device)

    check_accuracy_on_classes(loader=loader, model=model, device=device)
import os
import glob
import numpy as np
import torch
from deepcare.utils import msa

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from deepcare.dataset import generate_center_base_train_images


if __name__ == "__main__":


    msa_data_path = "/home/mnowak/care/care-output/"
    msa_file_name = "humanchr1430covMSA_*"
    msa_file_name_pattern = os.path.join(msa_data_path, msa_file_name)
    msa_file_paths = glob.glob(msa_file_name_pattern)

    fastq_file_path = "/share/errorcorrection/datasets/artmiseqv3humanchr14"
    fastq_file_name = "humanchr1430cov_errFree.fq.gz"
    fastq_file = os.path.join(fastq_file_path, fastq_file_name)

    folder_name = "datasets/humanchr1430covMSA_center_base_dataset_w51_h100_n64000_not_human_readable"

    generate_center_base_train_images(
            msa_file_paths=msa_file_paths,
            ref_fastq_file_path=fastq_file,
            image_height=100,
            image_width=250,
            out_dir=folder_name,
            max_number_examples=160000,
            human_readable=False,
            verbose=True
        )
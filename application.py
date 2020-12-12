import os
import glob
import numpy as np
import torch
from deepcare.utils import msa

msa_data_path = "/home/mnowak/care/care-output/"
msa_file_name = "humanchr1430covMSA_"

fastq_file_path = "/share/errorcorrection/datasets/artmiseqv3humanchr14" # /humanchr1430cov_errFree.fq"
fastq_file_name = "humanchr1430cov_errFree.fq.gz"
fastq_file = os.path.join(fastq_file_path, fastq_file_name)


if __name__ == "__main__":
    
    gen = msa.make_examples_from_file(
        msa_file=os.path.join(msa_data_path, msa_file_name+"1"),
        reference_fastq_file=fastq_file,
        image_height=50,
        image_width=25
    )
    print("Done creating the generator...")

    for i in gen:
        msa, label = i
        #save_msa_as_image(msa, "test_msa.png")
        print(label)
        break
    
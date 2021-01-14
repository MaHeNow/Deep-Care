import os
from glob import glob

from deepcare.data import bulk_generate_center_base_train_images

if __name__ == "__main__":

    msa_data_path = "/home/mnowak/care/humanchr1430covMSA/"
    msa_file_name = "part_*"
    msa_files = glob(os.path.join(msa_data_path, msa_file_name))
    msa_files = msa_files[:1]

    fastq_file_path = "/share/errorcorrection/datasets/artmiseqv3humanchr14"
    fastq_file_name = "humanchr1430cov_errFree.fq.gz"
    fastq_file = os.path.join(fastq_file_path, fastq_file_name)

    folder_name = "datasets/humanchr1430covMSA_center_base_dataset_w51_h100_human_readable"

    bulk_generate_center_base_train_images(
        msa_file_paths=msa_files,
        ref_fastq_file_path=fastq_file,
        image_height= 100,
        image_width= 51,
        out_dir=folder_name,
        workers=1,
        human_readable=True,
        verbose=True
    )
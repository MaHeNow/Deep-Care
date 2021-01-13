import os

from deepcare.data import generate_center_base_train_images_parallel

if __name__ == "__main__":

    msa_data_path = "/home/mnowak/care/care-output2/"
    msa_file_name = "humanchr1430covMSA_5"
    msa_file = os.path.join(msa_data_path, msa_file_name)

    fastq_file_path = "/share/errorcorrection/datasets/artmiseqv3humanchr14"
    fastq_file_name = "humanchr1430cov_errFree.fq.gz"
    fastq_file = os.path.join(fastq_file_path, fastq_file_name)

    folder_name = "datasets/humanchr1430covMSA_non_hq_part5_center_base_dataset_w51_h100_n512000_not_human_readable"

    generate_center_base_train_images_parallel(
        msa_file_path=msa_file,
        ref_fastq_file_path=fastq_file,
        image_height= 100,
        image_width= 51,
        out_dir=folder_name,
        max_num_examples=512000,
        workers=40,
        human_readable=False,
        verbose=True
    )
import os

from deepcare.data import generate_center_base_train_images_parallel, generate_center_base_train_images_parallel_v2,  generate_center_base_train_images_parallel_v3

if __name__ == "__main__":

    msa_data_path = "/home/mnowak/care/care-output2/"
    msa_file_name = "humanchr1430covMSA_1"
    msa_file = os.path.join(msa_data_path, msa_file_name)

    fastq_file_path = "/share/errorcorrection/datasets/artmiseqv3humanchr14"
    fastq_file_name = "humanchr1430cov_errFree.fq.gz"
    fastq_file = os.path.join(fastq_file_path, fastq_file_name)

    folder_name = "datasets/parallel_test"

    generate_center_base_train_images_parallel_v3(
        msa_file_path=msa_file,
        ref_fastq_file_path=fastq_file,
        image_height=100,
        image_width=51,
        out_dir=folder_name,
        workers=2,
        human_readable=True,
        verbose=True
    )
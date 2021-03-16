import os

from deepcare.data import generate_examples

if __name__ == "__main__":
    msa_data_path = "/home/mnowak/care/artmiseqv3humanchr1430covMSA_3"
    msa_file_paths = glob(os.path.join(msa_data_path, "part_*"))

    fastq_file_path = "/share/errorcorrection/datasets/artmiseqv3humanchr14"
    fastq_file_name = "humanchr1430cov_3_errFree.fq"
    fastq_file = os.path.join(fastq_file_path, fastq_file_name)

    folder_name = f"datasets/w451_h221/humanchr1430cov_3"

    generate_examples(
        msa_file_paths=msa_file_paths,
        ref_fastq_file_path=fastq_file,
        image_height= 221,
        image_width=451,
        out_dir=folder_name
    )
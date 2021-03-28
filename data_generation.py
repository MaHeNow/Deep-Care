import os
from glob import glob
from deepcare.data import generate_examples

if __name__ == "__main__":
    msa_data_path = "/home/mnowak/care/test/artmiseqv3humanchr1430covMSA_2"
    msa_file_paths = glob(os.path.join(msa_data_path, "part_*"))

    fastq_file_path = "/share/errorcorrection/datasets/artmiseqv3humanchr14"
    fastq_file_name = "humanchr1430cov_2.fq"
    ref_fastq_file_name = "humanchr1430cov_2_errFree.fq"
    
    fastq_file = os.path.join(fastq_file_path, fastq_file_name)
    ref_fastq_file = os.path.join(fastq_file_path, ref_fastq_file_name)

    folder_name = f"/home/mnowak/data/quality_datasets/w221_h221/humanchr1430cov_2"

    generate_examples(
        msa_file_paths=msa_file_paths,
        ref_fastq_file_path=ref_fastq_file,
        fastq_file_path=fastq_file,
        image_height= 221,
        image_width=221,
        out_dir=folder_name,
        workers=8
    )
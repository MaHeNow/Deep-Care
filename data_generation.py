import os
from glob import glob
from deepcare.data import generate_examples

if __name__ == "__main__":
    msa_data_path = "/home/mnowak/care/test/artmiseqv3humanchr1430covMSA_5"
    msa_file_paths = glob(os.path.join(msa_data_path, "part_*"))

    fastq_file_path = "/share/errorcorrection/datasets/artmiseqv3humanchr14"
    fastq_file_name = "humanchr1430cov_5.fq"
    ref_fastq_file_name = "humanchr1430cov_5_errFree.fq"

    fastq_file = os.path.join(fastq_file_path, fastq_file_name)
    ref_fastq_file = os.path.join(fastq_file_path, ref_fastq_file_name)

    folder_name = f"/home/mnowak/data/fresh_start/binary_quality_balanced_datasets/w1_h151/humanchr1430cov_5_parts_0_39"

    generate_examples(
        msa_file_paths=msa_file_paths,
        ref_fastq_file_path=ref_fastq_file,
        fastq_file_path=fastq_file,
        image_height=151,
        image_width=1,
        out_dir=folder_name,
        use_quality=True,
        extra_balancing=True,
        human_readable=True,
        workers=8
    )

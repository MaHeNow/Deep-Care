import os

from deepcare.data import generate_center_base_train_images_parallel

if __name__ == "__main__":

    parts = [i for i in range(8, 12)]

    for part in parts:
        msa_data_path = "/home/mnowak/care/arthiseq2000melanogaster30covMSA"
        msa_file_name = f"part_{part}"
        msa_file = os.path.join(msa_data_path, msa_file_name)

        fastq_file_path = "/share/errorcorrection/datasets/arthiseq2000melanogaster"
        fastq_file_name = "melanogaster30cov_errFree.fq"
        fastq_file = os.path.join(fastq_file_path, fastq_file_name)

        folder_name = f"datasets/w51_h100/arthiseq2000melanogaster30covMSA/part_{part}"

        generate_center_base_train_images_parallel(
            msa_file_path=msa_file,
            ref_fastq_file_path=fastq_file,
            image_height= 100,
            image_width=51,
            out_dir=folder_name,
            max_num_examples=None,
            workers=10,
            human_readable=True,
            verbose=True
        )
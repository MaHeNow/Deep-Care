import os

from nucleus.io import fastq


if __name__ == "__main__":

    fastq_file_path = "/share/errorcorrection/datasets/artmiseqv3humanchr14"
    fastq_file_name = "humanchr1430cov_errFree.fq.gz"
    fastq_file = os.path.join(fastq_file_path, fastq_file_name)

    with fastq.FastqReader(fastq_file) as reader:
        for i, read in enumerate(reader):
            print(read.quality)
            if i >= 10000-1:
                break
        
    

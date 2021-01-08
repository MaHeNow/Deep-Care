from multiprocessing import Pool
from deepcare.utils.msa import create_msa, nuc_to_index, crop_msa
import torch
import os
import time
import multiprocessing
from nucleus.io import fastq
from itertools import repeat

def some(chunks, image_width, image_height, ref_reads):
    return image_height


def parallel_func_2_pool(chunk, image_width, image_height, human_readable, allowed_bases):

    with fastq.FastqReader(ref_fastq_file_path) as reader:
        ref_reads = [read for read in reader]

    file_ = open(msa_file_path, "r")
    lines = [line.replace('\n', '') for line in file_]

    examples = {i : [] for i in allowed_bases}
    erroneous_examples = {i : [] for i in allowed_bases}

    for header_line_number in chunk:

        number_rows, number_columns, anchor_in_msa, anchor_in_file, high_quality = [int(i) for i in lines[header_line_number].split(" ")]
        start_height = header_line_number+1
        end_height = header_line_number+number_rows+1

        msa_lines = lines[start_height:end_height]
        anchor_column_index, anchor_sequence = msa_lines[anchor_in_msa].split(" ")

        # Get the reference sequence
        reference = ref_reads[anchor_in_file].sequence
        
        # Create a pytorch tensor encoding the msa from the text file
        msa = create_msa(msa_lines, number_rows, number_columns)

        # Go over entire anchor sequence and compare it with the reference
        for i, (b, rb) in enumerate(zip(anchor_sequence, reference)):

            center_index = i + int(anchor_column_index)
            column = msa[:, :, center_index]
            base_counts = torch.sum(column, dim=1)
            consensus_bases = torch.where(base_counts == torch.max(base_counts))
            if nuc_to_index[b] in consensus_bases:
                continue

            # Crop the MSA around the currently centered base
            cropped_msa = crop_msa(msa, image_width, image_height, center_index+1, anchor_in_msa)
            
            label = rb

            if b == label:
                examples[label].append(cropped_msa)
            else:
                erroneous_examples[label].append(cropped_msa)

    return (examples, erroneous_examples)


if __name__ == "__main__":

    msa_data_path = "/home/mnowak/care/care-output2/"
    msa_file_name = "humanchr1430covMSA_1"
    msa_file = os.path.join(msa_data_path, msa_file_name)

    fastq_file_path = "/share/errorcorrection/datasets/artmiseqv3humanchr14"
    fastq_file_name = "humanchr1430cov_errFree.fq.gz"
    fastq_file = os.path.join(fastq_file_path, fastq_file_name)

    folder_name = "datasets/parallel_test"

    msa_file_path=msa_file
    ref_fastq_file_path=fastq_file
    image_height=100
    image_width=51
    out_dir=folder_name
    workers=40
    human_readable=True
    verbose=True

    allowed_bases = "ACGT"

    examples = {i : [] for i in allowed_bases}
    erroneous_examples = {i : [] for i in allowed_bases}

    num_processes = min(multiprocessing.cpu_count()-1, workers) or 1
    if verbose:
        print(f"The data will be generated using {num_processes} parallel processes.")


    # -------------- Reading the MSA file --------------------------------------
    if verbose:
        print(f"Reading file {msa_file_path}")
    start = time.time()

    file_ = open(msa_file_path, "r")
    lines = [line.replace('\n', '') for line in file_]
    
    end = time.time()
    duration = end - start

    if verbose:
        print(f"Done. Reading the MSA file took {duration} seconds.")


    # -------------- Counting MSAs ---------------------------------------------
    if verbose:
        print("Counting the MSAs...")
    start = time.time()
    
    reading = True
    header_line_number = 0
    header_line_numbers = []
    #raw_msas = []
    million_counter = 0

    while reading:

        if header_line_number >= len(lines):
            reading = False
            continue

        if header_line_number >= million_counter*1000000:
            print(f"Currently at line {header_line_number}")
            million_counter += 1

        number_rows, number_columns, anchor_in_msa, anchor_in_file, high_quality = [int(i) for i in lines[header_line_number].split(" ")]
        
        # Only append the MSA if it is not of high quality
        if not high_quality:
            header_line_numbers.append(header_line_number)
        
        header_line_number += number_rows + 1

    end = time.time()
    duration = end - start

    if verbose:
        print("Done")
        print(f"There were {len(header_line_numbers)} many MSAs found. Counting the MSAs took {duration} seconds") 


    # -------------- Distributing work to multiple processes -------------------
    if verbose:
        print("Creating and saving examples...")
    start = time.time()

    header_line_numbers = header_line_numbers[:800]

    chunk_length = len(header_line_numbers) // num_processes
    chunks = [header_line_numbers[i:i+chunk_length] for i in range(0, len(header_line_numbers), chunk_length)]
    results = []
    print(f"Number chunks: {len(chunks)}")
    with Pool(num_processes) as pool:
        for chunk in chunks:
            args = [chunk, image_width, image_height, human_readable, allowed_bases]
            results.append(pool.apply_async(parallel_func_2_pool, args))
    
        pool.close()
        pool.join()
    
    for result in results:
        process_examples, process_erroneous_examples = result.get(timeout=10)
        for key in process_examples:
            examples[key] += process_examples[key]
            erroneous_examples[key] += process_erroneous_examples[key]

    end = time.time()
    duration = end - start

    if verbose:
        print(f"Done. Creating the examples took {duration} seconds.")

    for key in examples:
        print(key, " : ", len(examples[key]))

    for key in erroneous_examples:
        print(key, " : ", len(examples[key]))
import os 
import time
import numba as nb
from numba import jit, njit, prange, types
import numpy as np
from nucleus.io import fastq
from tqdm import tqdm
import ray


nuc_to_index = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3
    }
allowed_bases = np.array(["A", "C", "G", "T"])


def read_reference_file(ref_file_path):
    
    print("Reading the reference file... ")
    start = time.time()
    
    with fastq.FastqReader(ref_file_path) as reader:
        ref_reads = np.array([read.sequence for read in reader])
    
    end = time.time()
    duration = end - start

    print(f"Done. Reading the refernce file took {duration} seconds.")

    return ref_reads


def read_msa_file(msa_file_path):
    
    print(f"Reading file {msa_file_path}")
    start = time.time()

    file_ = open(msa_file_path, "r")
    lines = np.array([line.replace('\n', '') for line in file_])
    
    end = time.time()
    duration = end - start

    print(f"Done. Reading the MSA file took {duration} seconds.")

    return lines


def get_header_line_numbers(lines: np.ndarray) -> np.ndarray:

    print("Counting the MSAs...")
    start = time.time()
    
    reading = True
    header_line_number = 0
    header_line_numbers = []

    while reading:

        if header_line_number >= len(lines):
            reading = False
            continue

        number_rows, number_columns, anchor_in_msa, anchor_in_file, high_quality = [int(i) for i in lines[header_line_number].split(" ")]
        
        # Only append the MSA if it is not of high quality
        if not high_quality:
            header_line_numbers.append(header_line_number)
        
        header_line_number += number_rows + 1

    header_line_numbers = np.array(header_line_numbers)

    end = time.time()
    duration = end - start

    print("Done")
    print(f"There were {len(header_line_numbers)} many MSAs found. Counting the MSAs took {duration} seconds")

    return header_line_numbers

def create_msas(lines, header_line_numbers):

    # Arrays holding the MSAs and additoinal data
    msas = np.empty(header_line_numbers.size, dtype=object)
    anchors = []
    anchors_in_msa = np.zeros(header_line_numbers.size, dtype=np.int)
    anchor_column_indices = np.zeros(header_line_numbers.size, dtype=np.int)
    anchors_in_file = np.zeros(header_line_numbers.size, dtype=np.int)

    for i in prange(header_line_numbers.size):

        # Get the current header line number
        header_line_number = header_line_numbers[i]
        # Retrieve info about the MSA from the header line
        number_rows, number_columns, anchor_in_msa, anchor_in_file, _ = [int(j) for j in lines[header_line_number].split(" ")]
        start = header_line_number+1
        end = start+number_rows
        msa_lines = lines[start:end]
        anchor_column_index, anchor = msa_lines[anchor_in_msa].split(" ")
        
        # Create the MSA
        msas[i] = create_msa(msa_lines, number_rows, number_columns)
        anchors.append(anchor)
        anchors_in_msa[i] = anchor_in_msa
        anchors_in_file[i] = anchor_in_file
        anchor_column_indices[i] = int(anchor_column_index)
    
    return msas, np.array(anchors), anchors_in_msa, anchor_column_indices, anchors_in_file


def create_msa(msa_lines, number_rows, number_columns):

    # Initialize the MSA
    msa_shape = (4, number_rows, number_columns)
    msa = np.zeros(msa_shape, dtype=np.short)

    for i in prange(msa_lines.size):

        # Fill the MSA into the one-hot encoded MSA
        line = msa_lines[i]
        column_index, sequence = line.split(" ")
        column_index = int(column_index)
        for j in prange(len(sequence)):
            nucleotide = sequence[j]
            msa[nuc_to_index[nucleotide], i, column_index+j] = 1

    return msa


def find_positions(msas, anchors, anchors_in_msa, anchor_column_indices, anchros_in_file, ref_reads, allowed_bases, nuc_to_index):

    # Create Datastructure for holding positions
    consensus_examples = [[] for b in allowed_bases]
    non_consensus_examples = [[] for b in allowed_bases]

    for i in prange(msas.size):

        # Get MSA data
        msa = msas[i]
        anchor = anchors[i]
        anchor_in_msa = anchors_in_msa[i]
        anchor_column_index = anchor_column_indices[i]
        reference = ref_reads[anchros_in_file[i]]

        for j in range(len(reference)):
            
            # Compare every position in the anchor with the reference sequence
            base = anchor[j]
            ref_base = reference[j]

            if not ref_base in allowed_bases:
                continue
            
            center_index = j + anchor_column_index
            column = msa[:,:,center_index]
            base_counts = np.sum(column, axis=1)
            consensus_bases = np.where(base_counts == np.max(base_counts))[0]
            base_index = nuc_to_index[base]
            ref_base_index = nuc_to_index[ref_base]
            
            # Ignore cases where the base in the anchor is the consensus
            if base_index in consensus_bases:
                continue
            
            if ref_base_index in consensus_bases:
                consensus_examples[ref_base_index].append([i, anchor_in_msa, center_index])
            else:
                non_consensus_examples[ref_base_index].append([i, anchor_in_msa, center_index])
        
    min_count_examples = min(
            min([len(base_list) for base_list in consensus_examples]),
            min([len(base_list) for base_list in non_consensus_examples])
        )
    
    positions = np.array([
            [base_list[:min_count_examples] for base_list in consensus_examples],
            [base_list[:min_count_examples] for base_list in non_consensus_examples]
        ]) 

    return positions


def find_chunk_positions(start_i, end_i, msas, anchors, anchors_in_msa, anchor_column_indices, anchros_in_file, ref_reads, allowed_bases, nuc_to_index):

    # Create Datastructure for holding positions
    consensus_examples = [[] for b in allowed_bases]
    non_consensus_examples = [[] for b in allowed_bases]

    


def generate_center_base_train_images_numba(msa_file_path, ref_fastq_file_path, image_height, image_width, out_dir, max_num_examples=None, workers=1, human_readable=False, verbose=False):
    
    nuc_to_index = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3
    }
    allowed_bases = np.array(["A", "C", "G", "T"])
    
    target_num_examples = -1
    if max_num_examples != None:
        target_num_examples = max_num_examples//(2*len(allowed_bases))

    # -------------- Reading the reference file --------------------------------
    ref_reads = read_reference_file(ref_fastq_file_path)
    
    # -------------- Reading the MSA file --------------------------------------
    lines = read_msa_file(msa_file_path)
    
    # -------------- Counting MSAs ---------------------------------------------
    header_line_numbers = get_header_line_numbers(lines)
    header_line_numbers = header_line_numbers[:1000]
    
    start = time.time()
    msas, anchors, anchors_in_msa, anchor_column_indices, anchros_in_file = create_msas(lines, header_line_numbers)
    end = time.time()

    print(f"Oh hi there {end-start}")

    start = time.time()
    positions = find_positions(msas, anchors, anchors_in_msa, anchor_column_indices, anchros_in_file, ref_reads, allowed_bases, nuc_to_index)
    end = time.time()

    print(positions.shape)
    print(f"Oh hi there {end-start}")



if __name__ == "__main__":
    msa_data_path = "/home/mnowak/care/artmiseqv3humanchr1430covMSA"
    msa_file_name = f"part__{1}"
    msa_file = os.path.join(msa_data_path, msa_file_name)

    fastq_file_path = "/share/errorcorrection/datasets/artmiseqv3humanchr14"
    fastq_file_name = "humanchr1430cov_errFree.fq.gz"
    fastq_file = os.path.join(fastq_file_path, fastq_file_name)

    folder_name = f"datasets/w51_h100/balance_test_4_artmiseqv3humanchr1430covMSA/part_{1}"

    generate_center_base_train_images_numba(
        msa_file_path=msa_file,
        ref_fastq_file_path=fastq_file,
        image_height= 100,
        image_width=51,
        out_dir=folder_name,
        max_num_examples=None,
        workers=5,
        human_readable=True,
        verbose=True
    )
    
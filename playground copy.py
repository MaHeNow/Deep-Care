import os 
import time
import numba as nb
from numba import jit, njit, prange, types
import numpy as np
from nucleus.io import fastq
from tqdm import tqdm
import ray
from PIL import Image
import multiprocessing
from multiprocessing import Pool
from math import ceil
from glob import glob
from random import shuffle


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

def create_msas(lines, header_line_numbers, file_name, nuc_to_index):

    # Arrays holding the MSAs and additoinal data
    msas = np.empty(header_line_numbers.size, dtype=object)
    anchors = []
    anchors_in_msa = np.zeros(header_line_numbers.size, dtype=np.int)
    anchor_column_indices = np.zeros(header_line_numbers.size, dtype=np.int)
    anchors_in_file = np.zeros(header_line_numbers.size, dtype=np.int)

    for i in tqdm(range(header_line_numbers.size), desc=f"Creating MSAs for file {file_name}"):

        # Get the current header line number
        header_line_number = header_line_numbers[i]
        # Retrieve info about the MSA from the header line
        number_rows, number_columns, anchor_in_msa, anchor_in_file, _ = [int(j) for j in lines[header_line_number].split(" ")]
        start = header_line_number+1
        end = start+number_rows
        msa_lines = lines[start:end]
        anchor_column_index, anchor = msa_lines[anchor_in_msa].split(" ")
        
        # Create the MSA
        msas[i] = create_msa(msa_lines, number_rows, number_columns, nuc_to_index)
        anchors.append(anchor)
        anchors_in_msa[i] = anchor_in_msa
        anchors_in_file[i] = anchor_in_file
        anchor_column_indices[i] = int(anchor_column_index)
    
    return msas, np.array(anchors), anchors_in_msa, anchor_column_indices, anchors_in_file


def create_msa(msa_lines, number_rows, number_columns, nuc_to_index):

    # Initialize the MSA
    msa_shape = (4, number_rows, number_columns)
    msa = np.zeros(msa_shape, dtype=np.uint8)

    for i in prange(msa_lines.size):

        # Fill the MSA into the one-hot encoded MSA
        line = msa_lines[i]
        column_index, sequence = line.split(" ")
        column_index = int(column_index)
        for j in prange(len(sequence)):
            nucleotide = sequence[j]
            msa[nuc_to_index[nucleotide], i, column_index+j] = 1

    return msa


def find_positions(msas, anchors, anchors_in_msa, anchor_column_indices, anchros_in_file, ref_reads, allowed_bases, nuc_to_index, file_name):

    # Create Datastructure for holding positions
    consensus_examples = [[] for b in allowed_bases]
    non_consensus_examples = [[] for b in allowed_bases]

    for i in tqdm(prange(msas.size), desc=f"Finding Positions in file {file_name}"):

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
                consensus_examples[ref_base_index].append([i, anchor_in_msa, center_index, ref_base_index, len(consensus_examples[ref_base_index]), 1])
            else:
                non_consensus_examples[ref_base_index].append([i, anchor_in_msa, center_index, ref_base_index, len(non_consensus_examples[ref_base_index]), 0])
        
    min_count_examples = min(
            min([len(base_list) for base_list in consensus_examples]),
            min([len(base_list) for base_list in non_consensus_examples])
        )
    
    positions = []
    for base_list in consensus_examples:
        shuffle(base_list)
        positions += base_list[:min_count_examples]
    for base_list in non_consensus_examples:
        positions += base_list[:min_count_examples]

    positions = np.array(positions) 

    return positions


def generate_and_save_images(positions, msas, width, height, output_path, allowed_bases, nuc_to_color, file_name):

    for pos in tqdm(positions, desc=f"Creating images for file {file_name}"):
        msa_i, anch_msa_i, c_i, ref_i, im_i, cons = pos
        new_msa = crop_msa_to_image(msas[msa_i], width, height, c_i, anch_msa_i, nuc_to_color)
        im = Image.fromarray(new_msa)
        annotation = "cons" if cons else "ncons"
        name = f"{allowed_bases[ref_i]}_{annotation}_{im_i}.png"
        im.save(os.path.join(output_path, name))


def crop_msa_to_image(msa, new_width, new_height, new_center_x, new_center_y, nuc_to_color):

    number_channels, number_rows, number_columns = msa.shape
    new_shape = (new_height, new_width, number_channels)
    cropped_msa = np.zeros(new_shape, dtype=np.uint8)

    offset_x = int(new_center_x - new_width/2)
    offset_y = int(new_center_y - new_height/2)

    for x in range(new_width):
        for y in range(new_height):
            point_x = x+offset_x
            point_y = y+offset_y

            if 0 <= point_x < number_columns and 0 <= point_y < number_rows:
                indices = msa[:, y+offset_y, x+offset_x].nonzero()[0]
                if indices.size > 0:
                    base_index = indices[0]
                    color = nuc_to_color[base_index]
                    cropped_msa[y, x, :] = color

    return cropped_msa


def generate_examples_from_file(msa_file_path, ref_reads, image_width, image_height, root_dir):

    nuc_to_index = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3
    }
    nuc_to_color = {
          0 : np.array([  0,   0, 255, 255], dtype=np.uint8), # A becomes blue
          1 : np.array([255,   0,   0, 255], dtype=np.uint8), # C becomes red
          2 : np.array([  0, 255,   0, 255], dtype=np.uint8), # G becomes green
          3 : np.array([255, 255,   0, 255], dtype=np.uint8)  # T becomes yellow
    }
    allowed_bases = np.array(["A", "C", "G", "T"])
    file_name = os.path.basename(msa_file_path)

    # -------------- Reading the MSA file --------------------------------------
    lines = read_msa_file(msa_file_path)
    
    # -------------- Counting MSAs ---------------------------------------------
    header_line_numbers = get_header_line_numbers(lines)
    # TODO: the following line only exists for debugging purposes
    #header_line_numbers = header_line_numbers[:1000]
    
    # -------------- Creating MSAs ---------------------------------------------
    msas, anchors, anchors_in_msa, anchor_column_indices, anchros_in_file = create_msas(lines, header_line_numbers, file_name, nuc_to_index)

    # -------------- Finding Example Positions ---------------------------------
    positions = find_positions(msas, anchors, anchors_in_msa, anchor_column_indices, anchros_in_file, ref_reads, allowed_bases, nuc_to_index, file_name)

    # -------------- Generating and Savign Images ------------------------------
    out_dir = os.path.join(root_dir, f"{file_name}_{len(positions)}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    generate_and_save_images(positions, msas, image_width, image_height, out_dir, allowed_bases, nuc_to_color, file_name)


def generate_examples(msa_file_paths, ref_fastq_file_path, image_width, image_height, out_dir, workers=10):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # -------------- Reading the reference file --------------------------------
    ref_reads = read_reference_file(ref_fastq_file_path)
    number_iterations = ceil(len(msa_file_paths) / workers)

    start = time.time()

    for i in range(number_iterations):
        processes = []

        num_files_left = len(msa_file_paths) - i*workers
        num_files_this_it = workers if num_files_left >= workers else num_files_left
        
        for j in range(i*workers, i*workers + num_files_this_it):
            path = msa_file_paths[j]
            args = (path, ref_reads, image_width, image_height, out_dir)
            p = multiprocessing.Process(target=generate_examples_from_file, args=args)
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
    
    print(f"Completely done after {time.time()-start} seconds")


if __name__ == "__main__":
    msa_data_path = "/home/mnowak/care/artmiseqv3humanchr1430covMSA_3"
    #msa_file_name = f"part__{1}"
    #msa_file_name_2 = f"part__{2}"
    #msa_file = os.path.join(msa_data_path, msa_file_name)
    #msa_file_2 = os.path.join(msa_data_path, msa_file_name_2)
    #msa_file_paths = [msa_file, msa_file_2]
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
    
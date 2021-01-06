import os
import glob
import numpy as np
import random
from PIL import Image

import torch
from nucleus.io import fastq

from torch.utils.data import Dataset

nuc_to_index = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3
}
nuc_to_vector = {
    "A": torch.tensor([1, 0, 0, 0], dtype=torch.float32),
    "C": torch.tensor([0, 1, 0, 0], dtype=torch.float32),
    "G": torch.tensor([0, 0, 1, 0], dtype=torch.float32),
    "T": torch.tensor([0, 0, 0, 1], dtype=torch.float32)
}

def create_msa(msa_lines, number_rows, number_columns):

    msa_shape = (4, number_rows, number_columns)
    msa = torch.zeros(msa_shape, dtype=torch.float32)

    for i, line in enumerate(msa_lines):

        column_index, sequence = line.split(" ")
        column_index = int(column_index)
        for j, nucleotide in enumerate(sequence):
            msa[nuc_to_index[nucleotide], i, column_index+j] = 1

    return msa


def crop_msa(msa, new_width, new_height, new_center_x, new_center_y):

    number_channels, number_rows, number_columns = msa.shape
    new_shape = (number_channels, new_height, new_width)
    cropped_msa = torch.zeros(new_shape, dtype=torch.float32)

    offset_x = int(new_center_x - new_width/2)
    offset_y = int(new_center_y - new_height/2)

    for x in range(new_width):
        for y in range(new_height):
            point_x = x+offset_x
            point_y = y+offset_y

            if 0 <= point_x < number_columns and 0 <= point_y < number_rows:
                cropped_msa[:, y, x] = msa[:, y+offset_y, x+offset_x]

    return cropped_msa


def get_middle_base(sequence):
    """
    Get a vector representing the center base of a sequence
    """
    return nuc_to_index[sequence[len(sequence)//2]]


def save_msa_as_image(msa, name, root_dir, human_readable=False):
    msa = msa_to_image(msa, human_readable=human_readable)
    im = Image.fromarray(msa)
    im.save(os.path.join(root_dir, name))


def msa_to_image(msa, human_readable=False):

    COLORS = {
          "blue"   : [0, 0, 255, 255],
          "red"    : [255, 0, 0, 255],
          "green"  : [0, 255, 0, 255],
          "yellow" : [255, 255, 0, 255]
    }

    number_channels, number_rows, number_columns = msa.shape
    new_shape = (number_rows, number_columns, number_channels)
    image_msa = np.full(new_shape, fill_value=0, dtype=np.uint8)

    for x in range(number_columns):
        for y in range(number_rows):
            
            msa_pixel = msa[:, y, x]
            if (torch.equal(msa_pixel ,torch.tensor([1, 0, 0, 0], dtype=torch.float))):
                image_msa[y, x, :] = COLORS["blue"] if human_readable else [255, 0, 0, 0] # A becomes blue
            elif (torch.equal(msa_pixel ,torch.tensor([0, 1, 0, 0], dtype=torch.float))):
                image_msa[y, x, :] = COLORS["red"] if human_readable else [0, 255, 0, 0] # C becomes red
            elif (torch.equal(msa_pixel ,torch.tensor([0, 0, 1, 0], dtype=torch.float))):
                image_msa[y, x, :] = COLORS["green"] if human_readable else [0, 0, 255, 0] # G becomes green
            elif (torch.equal(msa_pixel ,torch.tensor([0, 0, 0, 1], dtype=torch.float))):
                image_msa[y, x, :] = COLORS["yellow"] if human_readable else [0, 0, 0, 255] # T becomes yellow
            
    return image_msa


def count_examples_of_file(msa_file_path, ref_fastq_file_path):

    erroneous_examples = {
        "A" : 0,
        "C" : 0,
        "G" : 0,
        "T" : 0
    }

    examples = {
        "A" : 0,
        "C" : 0,
        "G" : 0,
        "T" : 0
    }

    # read the reference file
    with fastq.FastqReader(ref_fastq_file_path) as reader:
        ref_reads = [read for read in reader]

    # open the msa file
    file_ = open(msa_file_path, "r")
    lines = [line.replace('\n', '') for line in file_]
    print(f"Number lines to read: {len(lines)}")

    reading = True
    header_line_number = 0

    while reading:

        print(f"Currently at line {header_line_number}")
        if header_line_number >= len(lines):
                reading = False
                continue
        
        # Get relavant information of the MSA from one of many heades in the
        # file encoding the MSAs
        number_rows, number_columns, anchor_in_msa, anchor_in_file = [int(i) for i in lines[header_line_number].split(" ")]

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

            # Count how many bases are correct and incorrect
            if rb == b:
                examples[rb] += 1
            else:
                erroneous_examples[rb] += 1
        
        # Jump to next MSA
        header_line_number += number_rows + 1
        
    print("Done counting examples.")
    print("Examples with correct bases: ", examples)
    print("Examples with erroneus bases: ", erroneous_examples)

    return examples, erroneous_examples


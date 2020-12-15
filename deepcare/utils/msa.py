import os
import glob
import numpy as np
import random
from PIL import Image

import torch
from nucleus.io import fastq

from torch.utils.data import Dataset

nuc_to_channel = {"A": 0, "C": 1, "G": 2, "T": 3}
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


class MSADataset(Dataset):
    def __init__(self, msa_file_path, ref_fastq_path, image_height, image_width, number_examples=None, transform=None):
        self.msa_file_path = msa_file_path
        self.ref_fastq_path = ref_fastq_path
        self.image_height = image_height
        self.image_width = image_width
        self.number_examples = number_examples
        self.transform = transform
        self.examples, self.labels = self.make_examples_from_file()

    def make_examples_from_file(self):
        
        msas = []
        labels = []
        i = 0

        # open a fastq reader to generate a list of all reference reads
        print("Starting to read the reference fastq file...")
        with fastq.FastqReader(self.ref_fastq_path) as reader:
            ref_reads = [read for read in reader]
        print("Done reading the reference file.")

        # open the file and generate a list of its lines
        file_ = open(self.msa_file_path, "r")
        lines = [line.replace('\n', '') for line in file_]

        reading = True
        header_line_number = 0

        while reading:
            
            if i == self.number_examples:
                break

            # continue reading until the last header of an msa
            if header_line_number >= len(lines):
                reading = False 
                continue

            number_rows, number_columns, anchor_in_msa, anchor_in_file = [int(i) for i in lines[header_line_number].split(" ")]
            
            start_height = header_line_number+1
            end_height = header_line_number+number_rows+1

            msa_lines = lines[start_height:end_height]
            anchor_column_index, anchor_sequence= msa_lines[anchor_in_msa].split(" ")

            center_index = int(anchor_column_index) + len(anchor_sequence)/2

            # createa a numpy array encoding the msa from the text file
            msa = create_msa(msa_lines, number_rows, number_columns)
            cropped_msa = crop_msa(msa, self.image_width, self.image_height, center_index, anchor_in_msa)

            reference = ref_reads[anchor_in_file].sequence
            middle_base = get_middle_base(reference)
            
            header_line_number += number_rows + 1

            msas.append(cropped_msa)
            labels.append(middle_base)

            i += 1

        return torch.stack(msas), labels

    def __len__(self):
        return self.examples.shape[0]

    def __getitem__(self, index):
        msa = self.examples[index]
        label = self.labels[index]
        if self.transform:
            msa = self.transform(msa)

        return (msa, torch.tensor(label))


def make_examples_from_file(msa_file, reference_fastq_file, image_height, image_width, number_examples=None):
    """
    Takes in the path of a text file encoding MSAs.
    Returns a generator with the output (MSA, label)
    """
    
    msas = []
    labels = []
    i = 0

    # open a fastq reader to generate a list of all reference reads
    print("Starting to read the reference fastq file...")
    with fastq.FastqReader(reference_fastq_file) as reader:
        ref_reads = [read for read in reader]
    print("Done reading the reference file.")

    # open the file and generate a list of its lines
    file_ = open(msa_file, "r")
    lines = [line.replace('\n', '') for line in file_]

    reading = True
    header_line_number = 0

    while reading:
        
        if i == number_examples:
            break

        # continue reading until the last header of an msa
        if header_line_number >= len(lines):
            reading = False 
            continue

        number_rows, number_columns, anchor_in_msa, anchor_in_file = [int(i) for i in lines[header_line_number].split(" ")]
        
        start_height = header_line_number+1
        end_height = header_line_number+number_rows+1

        msa_lines = lines[start_height:end_height]
        anchor_column_index, anchor_sequence= msa_lines[anchor_in_msa].split(" ")

        center_index = int(anchor_column_index) + len(anchor_sequence)/2

        # createa a numpy array encoding the msa from the text file
        msa = create_msa(msa_lines, number_rows, number_columns)
        cropped_msa = crop_msa(msa, image_width, image_height, center_index, anchor_in_msa)

        reference = ref_reads[anchor_in_file].sequence
        middle_base = get_middle_base(reference)
        
        header_line_number += number_rows + 1

        msas.append(cropped_msa)
        labels.append(middle_base)

        i += 1
    
    return torch.stack(msas), torch.tensor(labels)


def create_msa(msa_lines, number_rows, number_columns):

    msa_shape = (4, number_rows, number_columns)
    msa = torch.zeros(msa_shape, dtype=torch.float32)

    for i, line in enumerate(msa_lines):

        column_index, sequence = line.split(" ")
        column_index = int(column_index)
        for j, nucleotide in enumerate(sequence):
            msa[nuc_to_channel[nucleotide], i, column_index+j] = 1

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


def save_msa_as_image(msa, name):
    print("Image msa shape: ", msa.shape)
    msa_image = msa_to_image(msa)
    im = Image.fromarray(msa_image)
    im.save(name)


def msa_to_image(msa):

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
                image_msa[y, x, :] = COLORS["blue"] # A becomes blue
            elif (torch.equal(msa_pixel ,torch.tensor([0, 1, 0, 0], dtype=torch.float))):
                image_msa[y, x, :] = COLORS["red"] # C becomes red
            elif (torch.equal(msa_pixel ,torch.tensor([0, 0, 1, 0], dtype=torch.float))):
                image_msa[y, x, :] = COLORS["green"] # G becomes green
            elif (torch.equal(msa_pixel ,torch.tensor([0, 0, 0, 1], dtype=torch.float))):
                image_msa[y, x, :] = COLORS["yellow"] # T becomes yellow
            
    return image_msa

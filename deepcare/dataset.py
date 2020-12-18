import os
import random

import torch
from torch.utils.data import Dataset
from nucleus.io import fastq
import pandas as pd
from PIL import Image

from utils.msa import create_msa, crop_msa, get_middle_base, nuc_to_index, save_msa_as_image


def generate_center_base_train_images(msa_file_paths, ref_fastq_file_path, image_height, image_width, out_dir, max_number_examples, human_readable=False, verbose=False):
    
    number_examples = 0

    erroneous_examples = {
        "A" : [],
        "C" : [],
        "G" : [],
        "T" : []
    }

    examples = {
        "A" : [],
        "C" : [],
        "G" : [],
        "T" : []
    }

    # Get the reference reads for the MSAs
    if verbose:
        print("Reading the reference file... ", end="")
    with fastq.FastqReader(ref_fastq_file_path) as reader:
        ref_reads = [read for read in reader]
    if verbose:
        print("Done")

    max_num_examples_reached = False

    # Loop over all given files
    for path in msa_file_paths:

        # Termination condition of outer loop
        if max_num_examples_reached:
            break

        print("Now reading file: ", path)
        
        # Open the file and get its lines
        file_ = open(path, "r")
        lines = [line.replace('\n', '') for line in file_]

        reading = True
        header_line_number = 0

        while reading:

            # Termination conditions for inner loop
            if max_num_examples_reached:
                break

            if header_line_number >= len(lines):
                reading = False
                continue
        
            max_possible_examples = min(
                    min([len(val) for key, val in examples.items()]),
                    min([len(val) for key, val in erroneous_examples.items()])
                )

            if verbose:
                print(f"{max_possible_examples*(2*len(examples.keys()))} out of {max_number_examples} created.")

            if max_possible_examples == max_number_examples //(2*len(examples.keys())):
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
 
                center_index = i + int(anchor_column_index)+1

                max_possible_examples = min(
                    min([len(val) for key, val in examples.items()]),
                    min([len(val) for key, val in erroneous_examples.items()])
                )

                # Final termination condition for the inner loop
                if max_possible_examples == max_number_examples //(2*len(examples.keys())):
                    reading = False
                    max_num_examples_reached = True
                    break
                
                # Add no more examples to a group of bases if we have enough examples
                # for that specific base
                if b == rb:
                    if len(examples[rb]) >= max_number_examples//(2*len(examples.keys())):
                        continue
                else:
                    if len(erroneous_examples[rb]) >= max_number_examples//(2*len(erroneous_examples.keys())):
                        continue

                # Crop the MSA round the currently centered base
                cropped_msa = crop_msa(msa, image_width, image_height, center_index, anchor_in_msa)
                label = rb

                # Decide whether to add the example to the list of erroneus or
                # correct examples
                if b == rb:
                    examples[label].append(cropped_msa)
                else:
                    erroneous_examples[label].append(cropped_msa)

            header_line_number += number_rows + 1
            
    if verbose:
        print("Done creating examples.")

    # After all examples are generated, save the images to a folder and create a
    # training csv file

    # Prepare the dataframe which will be exported as a csv for indexing
    train_df = pd.DataFrame(columns=["img_name","label"])
    names = list()
    labels = list()

    # Create the output folder for the datset
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if verbose:
        print("Saving examples... ", end="")

    # Save all MSAs of the example list and add the filename and label to the csv
    for label, msas in examples.items():
        for i, msa in enumerate(msas):
            file_name = label + "_" + str(i) + ".png"
            save_msa_as_image(msa, file_name, out_dir, human_readable=human_readable)
            names.append(file_name)
            labels.append(nuc_to_index[label])

    # Do the same for the erroneus examples
    for label, msas in erroneous_examples.items():
        for i, msa in enumerate(msas):
            file_name = label + "_err_" + str(i) + ".png"
            save_msa_as_image(msa, file_name, out_dir, human_readable=human_readable)
            names.append(file_name)
            labels.append(nuc_to_index[label])

    train_df["img_name"] = names
    train_df["label"] = labels

    # Save the csv        
    train_df.to_csv (os.path.join(out_dir, "train_labels.csv"), index = False, header=True)

    if verbose:
        print("Done")


class MSADataset(Dataset):

    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img = Image.open(os.path.join(self.root_dir, img_id))
        y_label = torch.tensor(float(self.annotations.iloc[index, 1]))

        if self.transform is not None:
            img = self.transform(img)

        return (img, y_label)


# Examples usage:
"""
if __name__ == "__main__":
    import glob

    msa_data_path = "/home/mnowak/care/care-output/"
    msa_file_name = "humanchr1430covMSA_*"
    msa_file_name_pattern = os.path.join(msa_data_path, msa_file_name)
    msa_file_paths = glob.glob(msa_file_name_pattern)

    fastq_file_path = "/share/errorcorrection/datasets/artmiseqv3humanchr14" # /humanchr1430cov_errFree.fq"
    fastq_file_name = "humanchr1430cov_errFree.fq.gz"
    fastq_file = os.path.join(fastq_file_path, fastq_file_name)

    folder_name = "datasets/center_base_dataset_w11_h100_n800_human_readable"

    generate_center_base_train_images(
            msa_file_paths=msa_file_paths,
            ref_fastq_file_path=fastq_file,
            image_height=100,
            image_width=11,
            out_dir=folder_name,
            max_number_examples=800,
            human_readable=True,
            verbose=True
        )
"""
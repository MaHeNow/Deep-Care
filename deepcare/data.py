import os
import random
import time
import glob

import torch
from torch.utils.data import Dataset
from nucleus.io import fastq
import multiprocessing
from multiprocessing import Pool
from itertools import repeat
import pandas as pd
from PIL import Image

from deepcare.utils.msa import create_msa, crop_msa, get_middle_base, nuc_to_index, save_msa_as_image


def generate_center_base_train_images_parallel(msa_file_path, ref_fastq_file_path, image_height, image_width, out_dir, max_num_examples=None, workers=1, human_readable=False, verbose=False):

    allowed_bases = "ACGT"
    targe_num_examples = -1
    if max_num_examples != None:
        target_num_examples = max_num_examples//(len(allowed_bases))
    examples = {i : [] for i in allowed_bases}

    num_processes = min(multiprocessing.cpu_count()-1, workers) or 1
    if verbose:
        print(f"The data will be saved using {num_processes} parallel processes.")
    
    # -------------- Reading the reference file --------------------------------
    if verbose:
        print("Reading the reference file... ")
    start = time.time()
    
    with fastq.FastqReader(ref_fastq_file_path) as reader:
        ref_reads = [read for read in reader]
    
    end = time.time()
    duration = end - start

    if verbose:
        print(f"Done. Reading the refernce file took {duration} seconds.")


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


    # -------------- Generating the examples -----------------------------------
    if verbose:
        print("Now generating examples...")

    target_examples_reached = False
    reading = True
    start = time.time()

    for header_line_number in header_line_numbers:

        target_examples_reached = min([len(l) for l in examples.values()]) == target_num_examples

        # Termination conditions for inner loop
        if target_examples_reached:
            reading = False
            break

        if header_line_number >= len(lines):
            reading = False
            break

        # Get relavant information of the MSA from one of many heades in the
        # file encoding the MSAs
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

            if len(examples[rb]) >= target_num_examples:
                continue

            center_index = i + int(anchor_column_index)
            column = msa[:, :, center_index]
            base_counts = torch.sum(column, dim=1)
            consensus_bases = torch.where(base_counts == torch.max(base_counts))[0]
            nuc_index = nuc_to_index[b] 
            if nuc_index in consensus_bases:
                continue

            # Crop the MSA round the currently centered base
            cropped_msa = crop_msa(msa, image_width, image_height, center_index+1, anchor_in_msa)
            label = rb

            examples[label].append((cropped_msa, len(examples[rb])))

    end = time.time()
    duration = end - start

    if verbose:
        print(f"Done. Creating the examples took {duration} seconds.")

    
    # -------------- Create the output folder for the datset -------------------
    # determine the minimum number of examples
    min_num_examples = min([len(item) for key, item in examples.items()])
    for key in examples:
        examples[key] = examples[key][:min_num_examples]

    if not os.path.exists(f"{out_dir}_n{min_num_examples}"):
        os.makedirs(out_dir)


    # -------------- Saving examples -------------------------------------------
    if verbose:
        print("Saving examples.")
    start = time.time()
    
    for base in allowed_bases:

        processes = []

        start = time.time()
        chunk_length =  min_num_examples//num_processes

        for i in range(0, min_num_examples, chunk_length):
            examples_chunk = examples[base][i:i+chunk_length]
            args = (base, examples_chunk, out_dir, human_readable)
            p = multiprocessing.Process(target=save_images, args=args)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        
        if verbose:
            print(f"Done saving the {base}s.")
    
    end = time.time()
    duration = end - start

    if verbose:
        print(f"Done. Savin took {duration} seconds.")

    
    # -------------- Creating the annotations file -----------------------------
    # After all examples are generated and saved, create the training csv file
    if verbose:
        print("Now creating the annotations file...")
    start = time.time()
    
    # Prepare the dataframe which will be exported as a csv for indexing
    train_df = pd.DataFrame(columns=["img_name","label"])
    names = list()
    labels = list()

    for b in allowed_bases:
        files = glob.glob(os.path.join(out_dir, f"{b.upper()}*"))
        for file_ in files:
            names.append(os.path.basename(file_))
            labels.append(nuc_to_index[b])

    train_df["img_name"] = names
    train_df["label"] = labels

    # Save the csv        
    train_df.to_csv (os.path.join(out_dir, "train_labels.csv"), index = False, header=True)

    end = time.time()
    duration = end - start

    if verbose:
        print(f"Done. Creating the annotation file took {duration} seconds.")


def save_images(label ,examples, out_dir, human_readable):

    for (msa, index) in examples:
        file_name = label + "_" + str(index) + ".png"
        save_msa_as_image(msa, file_name, out_dir, human_readable=human_readable)


def bulk_generate_center_base_train_images(msa_file_paths, ref_fastq_file_path, image_height, image_width, out_dir, workers=1, human_readable=False, verbose=False):
    
    num_processes = min(multiprocessing.cpu_count()-1, workers, len(msa_file_paths)) or 1
    #TODO: This doesnt work
    workers = num_processes // len(msa_file_paths)
    if verbose:
        print(f"The data will be saved using {num_processes} parallel processes.")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_paths = [os.path.join(out_dir, os.path.basename(path)) for path in msa_file_paths]

    start = time.time()

    with Pool(num_processes) as pool:
        args = zip(msa_file_paths, repeat(ref_fastq_file_path), repeat(image_height), repeat(image_width), out_paths, repeat(400), repeat(workers), repeat(human_readable), repeat(verbose))
        pool.starmap(generate_center_base_train_images_parallel, args)

    end = time.time()
    duration = end - start
    if verbose:
        print(f"The entire process took {duration} seconds.")

    pass


class MSADataset(Dataset):

    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform
        #print("len: ", len(self.annotations))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img = Image.open(os.path.join(self.root_dir, img_id))
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]), dtype=torch.long)

        if self.transform is not None:
            img = self.transform(img)

        return (img, y_label)

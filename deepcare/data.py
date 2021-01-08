import os
import random
import time
import glob

import torch
from torch.utils.data import Dataset
from nucleus.io import fastq
import multiprocessing
import pandas as pd
from PIL import Image

from deepcare.utils.msa import create_msa, crop_msa, get_middle_base, nuc_to_index, save_msa_as_image


def generate_center_base_train_images(msa_file_paths, ref_fastq_file_path, image_height, image_width, out_dir, max_number_examples, human_readable=False, verbose=False):
    
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
        print("Reading the reference file... ")
    with fastq.FastqReader(ref_fastq_file_path) as reader:
        ref_reads = [read for read in reader]
    if verbose:
        print("Done")

    max_num_examples_reached = False
    start = time.time()

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
            number_rows, number_columns, anchor_in_msa, anchor_in_file, high_quality = [int(i) for i in lines[header_line_number].split(" ")]

            if high_quality:
                continue

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

    end = time.time()
    duration = end - start

    train_df["img_name"] = names
    train_df["label"] = labels

    # Save the csv        
    train_df.to_csv (os.path.join(out_dir, "train_labels.csv"), index = False, header=True)

    if verbose:
        print(f"Done. The process took {duration} seconds.")


def generate_center_base_train_images_parallel(msa_file_path, ref_fastq_file_path, image_height, image_width, out_dir, max_num_examples, human_readable=False, verbose=False):
    
    import multiprocessing
    manager = multiprocessing.Manager()

    global_start = time.time()

    allowed_bases = "ACGT"
    target_num_examples = max_num_examples //(2*len(allowed_bases))

    erroneous_examples_counter = manager.dict({i : 0 for i in allowed_bases})
    examples_counter = manager.dict({i : 0 for i in allowed_bases})

    # Create the output folder for the datset
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get the reference reads for the MSAs
    if verbose:
        print("Reading the reference file... ")
    start = time.time()
    with fastq.FastqReader(ref_fastq_file_path) as reader:
        ref_reads = [read for read in reader]
    end = time.time()
    duration = end - start
    if verbose:
        print(f"Done. Reading the reference file took {duration} seconds.")
    
    # Open the file and get its lines
    if verbose:
        print(f"Reading the file {msa_file_path}...")

    start = time.time()
    file_ = open(msa_file_path, "r")
    lines = [line.replace('\n', '') for line in file_]
    end = time.time()
    duration = end - start
    if verbose:
        print(f"Done. Reading took {duration} and the file has a total of {len(lines)} lines.")

    reading = True
    header_line_number = 0
    header_line_numbers = []
    million_counter = 0

    if verbose:
        print("Counting the MSAs...")

    start = time.time()

    while reading:

        if header_line_number >= len(lines):
            reading = False
            continue

        header_line_numbers.append(header_line_number)
        if header_line_number >= million_counter*1000000:
            print(f"Currently at line {header_line_number}")
            million_counter += 1

        number_rows, number_columns, anchor_in_msa, anchor_in_file = [int(i) for i in lines[header_line_number].split(" ")]
        header_line_number += number_rows + 1

    end = time.time()
    duration = end - start

    if verbose:
        print("Done")
        print(f"There were {len(header_line_numbers)} many MSAs found. Counting the MSAs took {duration} seconds")    

    if verbose:
        print("Creating and saving examples...")

    processes = []
    num_processes = multiprocessing.cpu_count()-1 or 1
    if verbose:
        print(f"The data will be generated using {num_processes} parallel processes.")

    start = time.time()
    chunk_length = len(header_line_numbers) // num_processes
    for i in range(0, len(header_line_numbers), chunk_length):
        chunk = header_line_numbers[i:i+chunk_length]
        args = (lines, chunk, image_width, image_height, ref_reads, out_dir, examples_counter, erroneous_examples_counter, target_num_examples, human_readable, allowed_bases)
        p = multiprocessing.Process(target=parallel_func, args=args)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    end = time.time()
    duration = end - start

    if verbose:
        print(f"Done. Creating and saving the examples took {duration} seconds.")

        print("This is the counting state")
        print("Normal ", examples_counter)
        print("Err, ", erroneous_examples_counter)

    if verbose:
        print("Now creating the annotations file...")

    # After all examples are generated and saved, create the training csv file
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

    global_end = time.time()
    global_duration = global_end - global_start

    if verbose:
        print(f"Done. Creating the annotations file took {duration} seconds.")
        print(f"The dataset is fully created. The entire process took {global_duration} seconds.")


def parallel_func(lines, header_line_numbers, image_width, image_height, ref_reads, out_dir, examples_counter, erroneous_examples_counter, target_num_examples, human_readable, allowed_bases):

    erroneous_examples = {i : [] for i in allowed_bases}
    examples = {i : [] for i in allowed_bases}
    done_searching = False

    for header_line_number in header_line_numbers:

        done_searching = min(min(examples_counter.values()), min(erroneous_examples_counter.values())) == target_num_examples
        if done_searching:
            break

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
            global_index = 0

            # Add no more examples to a group of bases if we have enough examples
            # for that specific base
            if b == rb:
                if examples_counter[rb] >= target_num_examples:
                    continue
                else:
                    global_index = examples_counter[rb]
                    examples_counter[rb] += 1
            else:
                if erroneous_examples_counter[rb] >= target_num_examples:
                    continue
                else:
                    global_index = erroneous_examples_counter[rb]
                    erroneous_examples_counter[rb] += 1

            # Crop the MSA around the currently centered base
            cropped_msa = crop_msa(msa, image_width, image_height, center_index, anchor_in_msa)
            label = rb

            # Decide whether to add the example to the list of erroneus or
            # correct examples
            if b == rb:
                examples[label].append((cropped_msa, global_index))
            else:
                erroneous_examples[label].append((cropped_msa, global_index))
            
        # Save all MSAs of the example list and add the filename and label to the csv
        for label, msa_index_pairs in examples.items():
            for (msa, index) in msa_index_pairs:
                file_name = label + "_" + str(index) + ".png"
                save_msa_as_image(msa, file_name, out_dir, human_readable=human_readable)

        # Do the same for the erroneus examples
        for label, msa_index_pairs in erroneous_examples.items():
            for (msa, index) in msa_index_pairs:
                file_name = label + "_err_" + str(index) + ".png"
                save_msa_as_image(msa, file_name, out_dir, human_readable=human_readable)


def generate_center_base_train_images_parallel_v2(msa_file_path, ref_fastq_file_path, image_height, image_width, out_dir, max_num_examples, workers=1, human_readable=False, verbose=False):
    
    import multiprocessing

    allowed_bases = "ACGT"
    target_num_examples = max_num_examples//(2*len(allowed_bases))

    examples = {i : [] for i in allowed_bases}
    erroneous_examples = {i : [] for i in allowed_bases}

    # Get the reference reads for the MSAs
    if verbose:
        print("Reading the reference file... ")
    with fastq.FastqReader(ref_fastq_file_path) as reader:
        ref_reads = [read for read in reader]
    if verbose:
        print("Done")


    # Open the file and get its lines
    if verbose:
        print(f"Reading file {msa_file_path}")
    file_ = open(msa_file_path, "r")
    lines = [line.replace('\n', '') for line in file_]
    if verbose:
        print("Done")


    if verbose:
        print("Now generating examples...")

    target_examples_reached = False
    reading = True
    header_line_number = 0
    start = time.time()

    while reading:

        target_examples_reached = min(
            min([len(l) for l in examples.values()]), 
            min([len(l) for l in erroneous_examples.values()])) == target_num_examples

        # Termination conditions for inner loop
        if target_examples_reached:
            reading = False
            break

        if header_line_number >= len(lines):
            reading = False
            break

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

            # Add no more examples to a group of bases if we have enough examples
            # for that specific base
            if b == rb:
                if len(examples[rb]) >= target_num_examples:
                    continue
            else:
                if len(erroneous_examples[rb]) >= target_num_examples:
                    continue

            # Crop the MSA round the currently centered base
            cropped_msa = crop_msa(msa, image_width, image_height, center_index, anchor_in_msa)
            label = rb

            # Decide whether to add the example to the list of erroneus or
            # correct examples
            if b == rb:
                examples[label].append((cropped_msa, len(examples[rb])))
            else:
                erroneous_examples[label].append((cropped_msa, len(erroneous_examples[rb])))

        header_line_number += number_rows + 1
    
    end = time.time()
    duration = end - start

    if verbose:
        print(f"Done. Creating the examples took {duration} seconds.")

    # Create the output folder for the datset
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # After all examples are generated, save the images to a folder and create a
    # training csv file
    num_processes = min(multiprocessing.cpu_count()-1, workers) or 1
    if verbose:
        print(f"The data will be saved using {num_processes} parallel processes.")

    start = time.time()
    for base in allowed_bases:

        processes = []

        start = time.time()
        chunk_length =  target_num_examples//num_processes

        for i in range(0, target_num_examples, chunk_length):
            examples_chunk = examples[base][i:i+chunk_length]
            err_examples_chunk = erroneous_examples[base][i:i+chunk_length]
            args = (base, examples_chunk, err_examples_chunk, out_dir, human_readable)
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

    if verbose:
        print("Now creating the annotations file...")

    # After all examples are generated and saved, create the training csv file
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


def save_images(label ,examples, erroneous_examples, out_dir, human_readable):

    for (msa, index) in examples:
        file_name = label + "_" + str(index) + ".png"
        save_msa_as_image(msa, file_name, out_dir, human_readable=human_readable)

    for (msa, index) in erroneous_examples:
        file_name = label + "_err_" + str(index) + ".png"
        save_msa_as_image(msa, file_name, out_dir, human_readable=human_readable)


def generate_center_base_train_images_parallel_v3(msa_file_path, ref_fastq_file_path, image_height, image_width, out_dir, workers=1, human_readable=False, verbose=False):

    allowed_bases = "ACGT"

    examples = {i : [] for i in allowed_bases}
    erroneous_examples = {i : [] for i in allowed_bases}

    num_processes = min(multiprocessing.cpu_count()-1, workers) or 1
    if verbose:
        print(f"The data will be generated using {num_processes} parallel processes.")


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


    # -------------- Distributing work to multiple processes -------------------
    if verbose:
        print("Creating and saving examples...")
    start = time.time()

    processes = []
    result_queue = multiprocessing.Queue()

    header_line_numbers = header_line_numbers[:2]

    chunk_length = len(header_line_numbers) // num_processes
    for i in range(0, len(header_line_numbers), chunk_length):
        chunk = header_line_numbers[i:i+chunk_length]
        args = (result_queue, lines, chunk, image_width, image_height, ref_reads, out_dir, human_readable, allowed_bases)
        p = multiprocessing.Process(target=parallel_func_2, args=args)
        p.start()
        processes.append(p)
    
    # merge examples
    while not result_queue.empty():
        process_examples, process_erroneus_examples = result_queue.get()
        for key in process_examples:
            examples[key].append(process_examples[key])
            erroneous_examples[key].append(process_erroneus_examples[key])

    # wait until all processes are done
    for p in processes:
        p.join()

    end = time.time()
    duration = end - start

    if verbose:
        print(f"Done. Creating and saving the examples took {duration} seconds.")


def parallel_func_2(results_queue, lines, chunk, image_width, image_height, ref_reads, out_dir, human_readable, allowed_bases):

    examples = {i : [] for i in allowed_bases}
    erroneus_examples = {i : [] for i in allowed_bases}

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
                erroneus_examples[label].append(cropped_msa)

    results_queue.put((examples, erroneus_examples))


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
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]), dtype=torch.long)

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

    folder_name = "datasets/humanchr1430covMSA_center_base_dataset_w51_h100_n64000_human_readable"

    generate_center_base_train_images(
            msa_file_paths=msa_file_paths,
            ref_fastq_file_path=fastq_file,
            image_height=100,
            image_width=250,
            out_dir=folder_name,
            max_number_examples=64000,
            human_readable=True,
            verbose=True
        )
"""
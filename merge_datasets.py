import os 
import shutil
import glob
import multiprocessing
import time
from tqdm import tqdm

import pandas as pd

from deepcare.utils.msa import nuc_to_index

if __name__ == "__main__":

    global_start = time.time()

    folder_paths = glob.glob("datasets/w51_h100/balance_test_4_artmiseqv3humanchr1430covMSA_Copy/*")

    result_folder_path = "datasets/w51_h100/balanced_artmiseqv3humanchr1430covMSA"
    #workers = 40

    bases = "ACGT"
    counter = {b : 0 for b in bases}

    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    else:
        print("There already is a folder at the result path")
        exit()

    names = []
    labels = []

    for folder in folder_paths:
        
        print(f"Now reading folder {folder}")
        start = time.time()
        images = glob.glob(os.path.join(folder, '*.png'))

        for i, image in tqdm(enumerate(images)):

            elements = os.path.basename(image).split("_")
            label = elements[0]
            if len(elements) == 4:
                annotation = f"{elements[1]}_{elements[2]}"
            else:
                annotation = elements[1]
            new_name = f"{label}_{annotation}_{counter[label]}.png"
            new_path = os.path.join(result_folder_path, new_name)
            shutil.move(image, new_path)

            counter[label] += 1
            names.append(new_name)
            labels.append(nuc_to_index[label])

        shutil.rmtree(folder)

        end = time.time()
        duration = end - start
        print(f"Done reading folder. Reading took {duration} seconds.")

    global_end = time.time()
    global_duration = global_end - global_start
    print(f"Done merging folders. The process took {duration} seconds.")

    train_df = pd.DataFrame(columns=["img_name","label"])

    train_df["img_name"] = names
    train_df["label"] = labels

    train_df.to_csv (os.path.join(result_folder_path, "train_labels.csv"), index = False, header=True)

    print("Done creating the index file.")
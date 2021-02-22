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

    folder_paths = [
        "/home/mnowak/deepcare/datasets/w51_h100/arthiseq2000athaliana30covMSACopy/part_8_n146476",
        "/home/mnowak/deepcare/datasets/w51_h100/arthiseq2000athaliana30covMSACopy/part_9_n154748",
        "/home/mnowak/deepcare/datasets/w51_h100/arthiseq2000athaliana30covMSACopy/part_10_n173860",
        "/home/mnowak/deepcare/datasets/w51_h100/arthiseq2000athaliana30covMSACopy/part_11_n160164",
        "/home/mnowak/deepcare/datasets/w51_h100/arthiseq2000athaliana30covMSACopy/part_12_n157776",
        "/home/mnowak/deepcare/datasets/w51_h100/arthiseq2000athaliana30covMSACopy/part_13_n177008",

        "/home/mnowak/deepcare/datasets/w51_h100/arthiseq2000elegans30covMSACopy/part_0_n111796",
        "/home/mnowak/deepcare/datasets/w51_h100/arthiseq2000elegans30covMSACopy/part_1_n133332",
        "/home/mnowak/deepcare/datasets/w51_h100/arthiseq2000elegans30covMSACopy/part_2_n127408",
        "/home/mnowak/deepcare/datasets/w51_h100/arthiseq2000elegans30covMSACopy/part_3_n123220",
        "/home/mnowak/deepcare/datasets/w51_h100/arthiseq2000elegans30covMSACopy/part_4_n128880",
        "/home/mnowak/deepcare/datasets/w51_h100/arthiseq2000elegans30covMSACopy/part_5_n99404",
        "/home/mnowak/deepcare/datasets/w51_h100/arthiseq2000elegans30covMSACopy/part_6_n132720",
        "/home/mnowak/deepcare/datasets/w51_h100/arthiseq2000elegans30covMSACopy/part_19_n114232"
    ]

    result_folder_path = "datasets/w51_h100/arthiseq2000AthalianaElegansMix_970032_970992_30covMSA"
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

            label = os.path.basename(image)[:1]
            new_name = f"{label}_{counter[label]}.png"
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
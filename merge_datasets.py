import os 
import shutil
import glob
import multiprocessing

import pandas as pd

from deepcare.utils.msa import nuc_to_index

if __name__ == "__main__":

    folder_paths = [
        "datasets/center_base/w51_h100/not_human_readable/melanogaster30covMSA/part_1_n16",
        "datasets/center_base/w51_h100/human_readable/melanogaster60covMSA/part_1_test_n16"
    ]

    result_folder_path = "datasets/merge_test"
    workers = 40

    bases = "ACGT"
    counter = {b : 0 for b in bases}

    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    else:
        print("There already is a folder at the result path")
        #exit()

    names = []
    labels = []

    for folder in folder_paths:
        
        images = glob.glob(os.path.join(folder, '*.png'))

        for i, image in enumerate(images):

            label = os.path.basename(image)[:1]
            new_name = f"{label}_{counter[label]}.png"
            new_path = os.path.join(result_folder_path, new_name)
            shutil.move(image, new_path)

            counter[label] += 1
            names.append(new_name)
            labels.append(nuc_to_index[label])
        
        shutil.rmtree(folder)

    train_df = pd.DataFrame(columns=["img_name","label"])

    train_df["img_name"] = names
    train_df["label"] = labels

    train_df.to_csv (os.path.join(result_folder_path, "train_labels.csv"), index = False, header=True)
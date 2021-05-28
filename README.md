# DeepCARE

## Prerequisites

### Operating System

`DeepCARE` needs be executed on a linux operating system. 

### Python Version

Use Python version `3.6.12` with Anaconda version `4.9.2`.

### Dependencies

To create a virtual environment with all the required dependencies and the correct python version, run the following command (replace `<env>` with the name you want to give the environment):

`conda create --name <env> --file requirements.txt python=3.6.12`

## Generating Datasets

Dataset generation works in two steps. First, generate a dataset using the `data_generation.py` script. It results in multiple folders with training images. Next, merge the newly created datasets using the `merge_datasets.py`script. It will create a single folder with all images of the specified folder paths. Additionally, it will create an index `.csv` file assigning a class to every image. This index file is required to load datasets during the training process.

## Training

Execute the `train.py` script with the desired hyperparamters.
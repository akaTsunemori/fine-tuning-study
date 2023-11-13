import os
import pandas as pd
import shutil
from pathlib import Path
from random import random


def copy_file(source_directory, destination_directory, filename):
    """
    Utility function used to copy a file from a source_directory to a destination_directory
    """
    destination_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_directory/filename, destination_directory/filename)


def organize_train_valid_dataset(root, labels, valid_probability=0.1):
    """
    Creates the train, train_valid and valid folders respecting PyTorch's ImageDataset structure, performing
    train/validation split based on the given percentage
    """
    source_directory = root/'original_train'
    with os.scandir(source_directory) as it:
        for entry in it:
            if entry.is_file():
                # The index is the name of the image except the extension
                img_index = entry.name.split('.')[0]
                # Find the class by looking up the index in the DF
                img_class = labels[labels.id == int(img_index)].label.values[0]
                # Randomly assign the image to the valid dataset with probability 'valid_probability'
                channel = Path('train') if random(
                ) > valid_probability else Path('valid')
                destination_directory = root/channel/img_class
                # Copy the image to either the train or valid folder, and also to the train_valid folder
                copy_file(source_directory, destination_directory, entry.name)
                copy_file(source_directory, root /
                          'train_valid'/img_class, entry.name)


def organize_test_dataset(root):
    """
    Creates the test folder respecting PyTorch's ImageDataset structure, using a dummy 'undefined' label
    """
    source_directory = root/'original_test'
    with os.scandir(source_directory) as it:
        for entry in it:
            if entry.is_file():
                # The index is the name of the image except the extension
                img_index = entry.name.split('.')[0]
                channel = Path('test')
                destination_directory = root/channel/'undefined'
                copy_file(source_directory, destination_directory, entry.name)


def organize_dataset():
    root = Path('./data')
    print('Reading trainLabels.csv')
    labels = pd.read_csv(root/'trainLabels.csv')
    print('Organazing datasets\' folder structures')
    valid_probability = 0.1
    organize_train_valid_dataset(root, labels, valid_probability)
    organize_test_dataset(root)

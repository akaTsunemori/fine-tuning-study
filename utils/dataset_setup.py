from os.path import exists

from extract import extract
from utils.organize_dataset import organize_dataset
from utils.split_dataset import split_dataset


def dataset_setup():
    if not exists('./data'):
        extract()
        organize_dataset()
        split_dataset()

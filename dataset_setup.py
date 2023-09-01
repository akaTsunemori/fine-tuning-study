from os.path import exists

from extract import extract
from organize_dataset import organize_dataset
from fix_dataset import fix_dataset


if not exists('./data'):
    extract()
    organize_dataset()
    fix_dataset()

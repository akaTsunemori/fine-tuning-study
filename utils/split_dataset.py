from os import listdir, mkdir, rename
from os.path import exists
from math import floor
from random import sample
from shutil import move, rmtree


def split_dataset():
    source  = './data/valid'
    destiny = './data/test_new'
    if not exists(destiny):
        mkdir(destiny)
    for folder in listdir(source):
        print(folder)
        if not exists(f'{destiny}/{folder}'):
            mkdir(f'{destiny}/{folder}')
        files = listdir(f'{source}/{folder}')
        count = len(files)
        to_move = sample(files, floor(count * 0.15))
        path_from = f'{source}/{folder}'
        path_to   = f'{destiny}/{folder}'
        for file in to_move:
            move(f'{path_from}/{file}', f'{path_to}/{file}')

    rmtree('./data/original_train')
    rmtree('./data/original_test')
    rmtree('./data/test')
    rename('./data/test_new', './data/test')

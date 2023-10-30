import py7zr
import shutil
from pathlib import Path


def extract():
    root = Path('./data')
    input_path = Path('./cifar-10')
    with py7zr.SevenZipFile(input_path/'train.7z', mode='r') as z:
        z.extractall(root)
    with py7zr.SevenZipFile(input_path/'test.7z', mode='r') as z:
        z.extractall(root)
    shutil.copy(input_path/'trainLabels.csv', root/'trainLabels.csv')
    (root/'train').rename(root/'original_train')
    (root/'test').rename(root/'original_test')

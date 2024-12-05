import os
from shutil import copyfile


def make_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

"""
Cette fonction sépare les images de CUB200 en un jeu d'entraînement et de test.

dataset_path: Path où se trouve les images de CUB200
train_path: path où sauvegarder le jeu d'entraînement
test_path: path où sauvegarder le jeu de test
"""


def separate_train_test(dataset_path, train_path, test_path):

    class_index = 1
    for classname in sorted(os.listdir(dataset_path)):
        if classname.startswith('.'):
            continue
        make_dir(os.path.join(train_path, classname))
        make_dir(os.path.join(test_path, classname))
        i = 0
        for file in sorted(os.listdir(os.path.join(dataset_path, classname))):
            if file.startswith('.'):
                continue
            file_path = os.path.join(dataset_path, classname, file)
            if i < 900:
                copyfile(file_path, os.path.join(test_path, classname, file))
            else:
                copyfile(file_path, os.path.join(train_path, classname, file))
            i += 1

        class_index += 1


# separate_train_test("./data/CUB_200_2011/CUB_200_2011/images","./data/train","./data/test")

separate_train_test("./data3/train","./data3/train/train","./data3/train/test")
from __future__ import print_function

import json
import cv2
from keras_preprocessing import image
import numpy as np
import os
from os.path import isfile, join
from os import listdir


class DirectoryHandler:

    def __init__(self, path):
        self.path = str(path)
        self.items_counts = self.__itemscounter(path)
        self.files_counts = self.__filescounter(path)
        self.folder_counts = self.__foldercounter(self.items_counts, self.files_counts)

    def __itemscounter(self, path):
        items = listdir(path)

        ds = ".DS_Store"
        try:
            ds_index = items.index(ds)
        except ValueError:
            pass
        else:
            items.pop(ds_index)
        return len(items)

    def __foldercounter(self, tot, files):
        if tot >= files:
            return tot - files
        else:
            "Errore"

    def __filescounter(self, path):

        onlyfiles = [".".join(f.split(".")[:-1]) for f in listdir(path) if isfile(join(path, f))]
        ds = ".DS_Store"
        try:
            ds_index = onlyfiles.index(ds)
        except ValueError:
            pass
        else:
            onlyfiles.pop(ds_index)

        self.files_counts = len(onlyfiles)
        self.onlyfiles = onlyfiles
        return self.files_counts

    """Conta numero di sotto cartelle"""

    def subfoldercount(self):
        return self.items - self.files_counts

    """ costruisce un dizionario le cui chiavi sono le sottocartelle, valore è il path delle sottocartelle"""

    def dict_folder_path_builder(self):
        path1 = self.path
        onlydir = [name for name in os.listdir(path1) if os.path.isdir(os.path.join(path1, name))]
        paths = {}
        for i in onlydir:
            newpath = path1 + "/" + str(i)
            paths[i] = newpath
        return paths

    def dict_file_path_builder(self):
        path1 = self.path
        onlydir = [name for name in os.listdir(path1) if os.path.isfile(os.path.join(path1, name))]
        paths = {}
        for i in onlydir:
            newpath = path1 + "/" + str(i)
            paths[i] = newpath
        return paths

    """Costruisce dei path"""

    @staticmethod
    def pathbuilder(path, folder):
        return path + "/" + folder


def load_img(path):
    return cv2.imread(path)


def smoothing(img, kernel):
    newimg = cv2.blur(img, kernel)
    return newimg


def random_shear(img, intensity_range=(-0.5, 0.5), u=0.5):
    if np.random.random() < u:
        sh = np.random.uniform(-intensity_range[0], intensity_range[1])
        img = image.apply_affine_transform(img, shear=sh)
    return img


def flip_image(image, ax):
    return cv2.flip(image, ax)


def random_grey(img, u=0.5):
    if np.random.random() < u:
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to grey
        gray = np.sum(img * coef, axis=2)
        img = np.dstack((gray, gray, gray))
    return img


def run_augmentation(folder_path):
    dh = DirectoryHandler(folder_path)

    files_dict = dh.dict_file_path_builder()

    print("Generazione augmentation per cartella " + folder_path + "...")
    for file in files_dict:

        if file != ".DS_Store":
            # caricamento img
            img = cv2.cvtColor(load_img(files_dict[file]), cv2.COLOR_BGR2RGB)

            # smoothing
            smoothed = smoothing(img, (5, 5))
            smoothedrgb = cv2.cvtColor(smoothed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(folder_path + "/" + "aug_smooth_" + file, smoothedrgb)

            # grey
            grey = random_grey(img)
            cv2.imwrite(folder_path + "/" + "aug_grey_" + file, grey)

            # flip 0
            flip = flip_image(img, 0)
            flipped = cv2.cvtColor(flip, cv2.COLOR_RGB2BGR)
            cv2.imwrite(folder_path + "/" + "aug_flipped0_" + file, flipped)

            # flip 1
            flip = flip_image(img, 1)
            flipped = cv2.cvtColor(flip, cv2.COLOR_RGB2BGR)
            cv2.imwrite(folder_path + "/" + "aug_flipped1_" + file, flipped)

            # shear
            shear = random_shear(img, intensity_range=(-10, 10))
            sheared = cv2.cvtColor(shear, cv2.COLOR_RGB2BGR)
            cv2.imwrite(folder_path + "/" + "aug_shear_" + file, sheared)
    # print("Generazione augmentation per cartella " + folder_path + " completata con successo!")


with open('config.json') as json_data_file:
    config = json.load(json_data_file)

output_dir = config["output_dir"]

dataset_path = output_dir + "/dataset"

dataset_train_path = dataset_path + "/train"
dataset_val_path = dataset_path + "/val"

dh = DirectoryHandler(dataset_train_path)

classes = dh.dict_folder_path_builder()

for c in classes:

    # esclude la prima classe, perché la più numerosa
    if (int(c) != 0):
        class_path = classes[c]
        run_augmentation(class_path)
print("")
print("Augmentation per train completata con successo!")

dh = DirectoryHandler(dataset_val_path)

classes = dh.dict_folder_path_builder()

for c in classes:

    # esclude la prima classe, perché la più numerosa
    if (int(c) != 0):
        class_path = classes[c]
        run_augmentation(class_path)
print("")
print("Augmentation per val completata con successo!")

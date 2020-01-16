from __future__ import print_function

import pandas as pd
import os
from os.path import isfile, join
from os import listdir
import json


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


# carica i valori di una colonna di un csv
def load_column_data(path, col):
    df = pd.read_csv(path)
    return df[col]


def moveTo(file, dir):
    os.rename(file, dir)


def organize(samples_path, label_csv_path, final_dest, percent):
    samples_path = samples_path + "/"
    print("Riorganizzazione dataset in corso...")
    # dh = directory.DirectoryHandler(samples_path)

    # print(dh.dict_file_path_builder())

    if not os.path.exists(final_dest):
        os.makedirs(final_dest)

    if not os.path.exists(final_dest + "/train"):
        os.makedirs(final_dest + "/train")

    if not os.path.exists(final_dest + "/val"):
        os.makedirs(final_dest + "/val")

    labels = load_column_data(label_csv_path, "level")
    nomi_esempi = load_column_data(label_csv_path, "image")

    ns = len(labels)

    for l in range(ns):
        percorso_esempio = samples_path + nomi_esempi[l] + ".jpeg"
        label = labels[l]

        if not os.path.exists(final_dest + "/train" + "/" + str(label)):
            os.makedirs(final_dest + "/train" + "/" + str(label))

        if not os.path.exists(final_dest + "/val" + "/" + str(label)):
            os.makedirs(final_dest + "/val" + "/" + str(label))

        moveTo(percorso_esempio, final_dest + "/train" + "/" + str(label) + "/" + nomi_esempi[l] + ".jpeg")
        #print(nomi_esempi[l] + ".jpeg" + " spostata")

    # creare cartelle se non esistono

    train_folder = final_dest + "/train"
    val_folder = final_dest + "/val"

    dh = DirectoryHandler(train_folder)

    print("Generazione validation test in corso...")
    # print(dh.dict_folder_path_builder())

    for f in dh.dict_folder_path_builder():
        dhf = DirectoryHandler(dh.dict_folder_path_builder()[f])

        ns = len(dhf.dict_file_path_builder())

        current_files_path = list(dhf.dict_file_path_builder().values())
        current_files_names = list(dhf.dict_file_path_builder().keys())

        nvs = int(round((ns / 100) * percent))

        for i in range(nvs):
            moveTo(current_files_path[i], val_folder + "/" + f + "/" + current_files_names[i])


with open('config.json') as json_data_file:
    config = json.load(json_data_file)

dataset_dir = config["dataset_dir"]
labels_path = config["labels_path"]
output_dir = config["output_dir"]
val_ratio = config["val_ratio"]

if not os.path.exists(output_dir + "/dataset"):
    os.makedirs(output_dir + "/dataset")

#crea cartella dove inserire i risultati dei vari training
if not os.path.exists(output_dir + "/results"):
    os.makedirs(output_dir + "/results")

"""riorganizza il dataset, creando una cartella dataset con dentro cartelle train e val. Il val viene creato 
prendendo una percentuale dalla cartella train """
organize(dataset_dir,
         labels_path,
         output_dir + "/" + "dataset",
         int(val_ratio))

print("Riorganizzazione completata con successo!")

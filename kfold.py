from __future__ import print_function

from sklearn.model_selection import KFold
import json
import os
from os.path import isfile, join
from os import listdir
from sklearn.utils import shuffle
import numpy as np
from shutil import copyfile



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


def moveTo(file, dir):
    os.rename(file, dir)


def getFileName(path):
    s = path
    return s.split("/")[8]


with open('config.json') as json_data_file:
    config = json.load(json_data_file)

output_dir = config["output_dir"]
kfold = config["fold"]

kf = KFold(n_splits=kfold)

if not os.path.exists(output_dir + "/kfold"):
    os.makedirs(output_dir + "/kfold")


kfold_path = output_dir + "/kfold"

print("Generazione " + str(kfold) + " fold...")
for i in range(kfold):
    os.makedirs(output_dir + "/kfold/" + str(i) + "fold")

print("Distribuzione in corso...")

tdh = DirectoryHandler(output_dir + "/dataset/train")

train_classes = tdh.dict_folder_path_builder()

train_samples = []
train_labels = []

for c in train_classes:
    # print(train_classes[c])
    cdh = DirectoryHandler(train_classes[c])

    sdh = cdh.dict_file_path_builder().values()
    # print(sdh)

    for s in sdh:
        train_samples.append(s)
        train_labels.append(c)

vdh = DirectoryHandler(output_dir + "/dataset/val")

val_classes = vdh.dict_folder_path_builder()

for c in val_classes:
    # print(train_classes[c])
    cdh = DirectoryHandler(val_classes[c])

    sdh = cdh.dict_file_path_builder().values()
    # print(sdh)

    for s in sdh:
        train_samples.append(s)
        train_labels.append(c)


samples = np.array(train_samples)
labels = np.array(train_labels)

X, y = shuffle(samples[:], labels[:])



for m in range(int(kfold)):
    folddir = output_dir + "/kfold/" + str(m) + "fold"

    traindir = folddir + "/train"
    valdir = folddir + "/val"

    os.makedirs(traindir)
    os.makedirs(valdir)

    for f in range(5):
        if not os.path.exists(traindir + "/" + str(f)):
            os.makedirs(traindir + "/" + str(f))
        if not os.path.exists(valdir + "/" + str(f)):
            os.makedirs(valdir + "/" + str(f))

i = 0

for train_index, test_index in kf.split(X):
    print("Generazione " + str(i) + "fold...")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    folddir = output_dir + "/kfold/" + str(i) + "fold"

    traindir = folddir + "/train"
    valdir = folddir + "/val"

    #os.makedirs(traindir)
    #os.makedirs(valdir)

    for f in range(5):
        if not os.path.exists(traindir + "/" + str(f)):
            os.makedirs(traindir + "/" + str(f))
        if not os.path.exists(valdir + "/" + str(f)):
            os.makedirs(valdir + "/" + str(f))

    for j in range(len(X_train)):
        copyfile(X_train[j], traindir + "/" + str(y_train[j]) + "/" + getFileName(X_train[j]))

    for z in range(len(X_test)):
        copyfile(X_test[z], valdir + "/" + str(y_test[z]) + "/" + getFileName(X_test[z]))

    i = i + 1

print("Generazione " + str(kfold) + "fold completata con successo!")
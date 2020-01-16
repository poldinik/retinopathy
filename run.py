import json
import os
import sys
import warnings

import cnn1
# import cnn2
import cnn2
import cnn3

with open('config.json') as json_data_file:
    config = json.load(json_data_file)

dataset_dir = config["dataset_dir"]
output_dir = config["output_dir"]
model = config["model"]
epoch = int(config["epoch"])
size = int(config["size"])
kfold = config["kfold"]
fold = int(config["fold"])
dense_level = int(config["dense_level"])
val_ratio = int(config["val_ratio"])
batch = int(config["batch"])

# run training
directory_dataset = output_dir + "/dataset"



if not sys.warnoptions:
    warnings.simplefilter("ignore")


if(size < 32):
    print("[INFO] Dimensione esempi troppo piccola. Modificare config.json con size pari a valori come 32, 64, 128, 256 e così via")
else:
    print("[INFO] Dimensione esempi: " + str(size) + "x" + str(size))
    if (kfold):

        import argparse

        ap = argparse.ArgumentParser()
        ap.add_argument("-f", "--fold", required=True, help="Indice della fold")
        args = vars(ap.parse_args())
        index = int(args["fold"])
        directory_dataset = output_dir + "/kfold/" + str(index) + "fold"
        print("[INFO] Esecuzione training su " + str(index) + "fold")
        if model == "standard":
            cnn1.run(epoch, size, batch, directory_dataset, output_dir + "/results")
        elif model == "fine":
            cnn2.run(epoch, size, batch, directory_dataset, output_dir + "/results")
        elif model == "regression":
            print("[INFO] Esecuzione pipeline apprendimento con modello di regressione")

            checkpoint = output_dir + "/checkpoint"
            if not os.path.exists(checkpoint):
                print("[INFO] Checkpoint non presente")
                print("[INFO] Creazione cartella checkpoint..")
                os.makedirs(checkpoint)
                cnn3.run(epoch, size, batch, directory_dataset, output_dir + "/results", dense_level, False, checkpoint)
            else:
                print("[INFO] Checkpoint presente (viene ricaricarto il modello trainato in precedenza)")
                cnn3.run(epoch, size, batch, directory_dataset, output_dir + "/results", dense_level, True, checkpoint)

    else:
        if model == "standard":
            cnn1.run(epoch, size, batch, directory_dataset, output_dir + "/results", dense_level)
        elif model == "fine":
            cnn2.run(epoch, size, batch, directory_dataset, output_dir + "/results", dense_level)
        elif model == "regression":
            print("[INFO] Esecuzione pipeline apprendimento con modello di regressione")


            checkpoint = output_dir + "/checkpoint"
            if not os.path.exists(checkpoint):
                print("[INFO] Checkpoint non presente")
                print("[INFO] Creazione cartella checkpoint..")
                os.makedirs(checkpoint)
                cnn3.run(epoch, size, batch, directory_dataset, output_dir + "/results", dense_level, False, checkpoint)
            else:
                print("[INFO] Checkpoint presente (viene ricaricarto il modello trainato in precedenza)")
                cnn3.run(epoch, size, batch, directory_dataset, output_dir + "/results", dense_level, True, checkpoint)

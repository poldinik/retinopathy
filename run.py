import json
import cnn1
# import cnn2
import cnn2

with open('config.json') as json_data_file:
    config = json.load(json_data_file)

dataset_dir = config["dataset_dir"]
output_dir = config["output_dir"]
model = config["model"]
epoch = int(config["epoch"])
size = int(config["size"])
kfold = config["kfold"]
fold = int(config["fold"])
val_ratio = int(config["val_ratio"])
batch = int(config["batch"])

# run training
directory_dataset = output_dir + "/dataset"

if (kfold):

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--fold", required=True, help="Indice della fold")
    args = vars(ap.parse_args())
    index = int(args["fold"])
    directory_dataset = output_dir + "/kfold/" + str(index) + "fold"
    print("Esecuzione training su " + str(index) + "fold")

else:
    print("Esecuzione senza kfold")
    if model == "standard":
        cnn1.run(epoch, size, batch, directory_dataset, output_dir + "/results")
    elif model == "fine":
        cnn2.run(epoch, size, batch, directory_dataset, output_dir + "/results")

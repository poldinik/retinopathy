import json
import cnn1
#import cnn2

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


if(kfold):

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--fold", required=True, help="Indice della fold")
    args = vars(ap.parse_args())
    index = int(args["fold"])
    print("Esecuzione training su " + str(index) + "fold")

else:
    print("Esecuzione senza kfold")
    # if(model == "standard"):
    #   cnn1.run(epoch, size, batch, output_dir + "/dataset", output_dir + "/results")
    # else if(model=="fine"):
    # #     cnn2.run(epoch, size, batch, output_dir + "/dataset", output_dir + "/results")
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import locale
import pandas as pd
from keras.utils import plot_model
import json

from regr_data import load_fundus_images
from regr_model import create_cnn

with open('config_regr.json') as json_data_file:
    config = json.load(json_data_file)

size = config["size"]
epochs = config["epochs"]
batch_size = config["batch_size"]
dataset = config["dataset"]
labels_path = config["labels_path"]
final_dest = config["final_dest"]


print("[INFO] caricamento labels...")

# size = 32
# epochs = 10
# batch_size = 8
# dataset = "/home/loretto/Desktop/resized_train_cropped/resized_train_cropped"
# labels_path = "/home/loretto/Desktop/trainLabels_cropped.csv"
#
# final_dest = "/home/loretto/Desktop/regression"


df = pd.read_csv(labels_path)


print("[INFO] caricamento esempi...")

images = load_fundus_images(df, dataset, size)
images = images / 255.0

print(images.shape)

split = train_test_split(df, images, test_size=0.20, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split


trainY = trainAttrX["level"]
testY = testAttrX["level"]

model = create_cnn(size, size, 3, regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt, metrics=["accuracy"])

print("[INFO] training model...")
history = model.fit(trainImagesX, trainY, validation_data=(testImagesX, testY),
	epochs=epochs, batch_size=batch_size)


#plot_model(model, to_file=final_dest + "/" + "model.png", show_shapes=True, show_layer_names=True)

print("[INFO] predizione...")
preds = model.predict(testImagesX)

diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)


mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)


locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. grado retinopatia: {}, std retinopatia: {}".format(
	df["level"].mean(),
	df["level"].std()))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))


df = pd.DataFrame(history.history)
df.to_excel(final_dest + "/" + "history.xls")

df = pd.DataFrame(preds)
df.to_excel(final_dest + "/" + "test_predicted" + ".xls")
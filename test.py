import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

from keras.preprocessing.image import ImageDataGenerator
import itertools
from keras.models import load_model
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


with open('config.json') as json_data_file:
    config = json.load(json_data_file)

output_dir = config["output_dir"]
size = int(config["size"])
batch_size = int(config["batch"])
regr_approximation = config["regr_approximation"]

rescale = 1. / 255
test_dir = output_dir + "/dataset/val"
target_size = (size, size)
checkpoint_folder = output_dir + "/checkpoint"

final_dest = output_dir + "/results"

model = load_model(checkpoint_folder + "/model.h5")

test_datagen = ImageDataGenerator(
        rescale=rescale,
        featurewise_center=True,
        shear_range=0.2,
        zoom_range=0.5,
        width_shift_range=0.5,
        horizontal_flip=True,
        rotation_range=130,
        zca_whitening=True)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    class_mode='sparse',
    batch_size=batch_size,
    color_mode="rgb",
    shuffle=True)

number_of_examples = len(test_generator.filenames)
number_of_generator_calls = np.math.ceil(number_of_examples / (1.0 * batch_size))

test_labels = []

print("[INFO] caricamento labels...")
for i in range(0, int(number_of_generator_calls)):
    test_labels.extend(np.array(test_generator[i][1]))

print("[INFO] salvataggio labels originali...")
test_labels = np.array(test_labels)
df = pd.DataFrame(test_labels)
testxls = "true_test_labels" + ".xls"
df.to_excel(final_dest + "/" + testxls)

print("[INFO] predizione...")
test_predictions = model.predict_generator(test_generator, steps=len(test_generator), verbose=1)

print("[INFO] salvataggio label predette...")
df = pd.DataFrame(test_predictions)
testxls = "predicted_test_labels" + ".xls"
df.to_excel(final_dest + "/" + testxls)

true = test_labels

if regr_approximation == "ceil":
    predicted = np.ceil(test_predictions)
elif regr_approximation == "modf":
    predicted = np.modf(test_predictions)[1]

cnf_matrix = confusion_matrix(true, predicted)
c = cohen_kappa_score(predicted, true)

print("[INFO] salvataggio matrice di confusione...")
df = pd.DataFrame(cnf_matrix)
matrix_path = final_dest + "/test_confusion_matrix" + ".xls"
df.to_excel(matrix_path)


acc = accuracy_score(true, predicted)

col0 = cnf_matrix

class_names = ["0", "1", "2", "3", "4"]

cm = []

nr = len(col0)

for c in range(nr):
    cm.append(list(df[c]))

cnf_matrix = np.array(cm).T

title = str(size) + "x" + str(size) + " Test Confusion Matrix (acc: " + str(round(acc, 2)) + ", K: " + str(round(c, 2)) + ")"

print("[INFO] creazione immagine matrice di confusione...")
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,title=title)
plt.savefig(final_dest + "/" + str(size) + "x" + str(size) + "_test_confusion_matrix" + ".png", dpi=500)
print("[INFO] fine...")
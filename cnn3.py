from __future__ import division

import numpy as np
import pandas as pd
import uuid

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import itertools
from matplotlib import rc
from keras.models import load_model
import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")


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


def save_confusion_matrix(matrix_path, name, title, final_dest):
    path = matrix_path

    df = pd.read_excel(path)

    col0 = df[0]

    class_names = ["0", "1", "2", "3", "4"]

    cm = []

    nr = len(col0)

    for c in range(nr):
        cm.append(list(df[c]))

    cnf_matrix = np.array(cm).T

    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                          title=title + " Confusion Matrix")

    plt.savefig(final_dest + "/" + name + "_cm" + ".png", dpi=500)


def save_history_images(history_path_file, name, final_dest):
    #rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    #rc('text', usetex=True)

    path = history_path_file

    df = pd.read_excel(path)

    nrows = len(df['acc'])

    acc = df['acc'][0:nrows]
    val_acc = df['val_acc'][0:nrows]

    loss = df['loss'][0:nrows]
    val_loss = df['val_loss'][0:nrows]

    plt.plot(acc, color="black")
    plt.plot(val_acc, linestyle='--', color="black")
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid(True)
    plt.savefig(final_dest + "/" + name + "_acc.png", dpi=500)

    # summarize history for loss
    plt.plot(loss, color="black")
    plt.plot(val_loss, linestyle='--', color="black")
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid(True)
    plt.savefig(final_dest + "/" + name + "_loss.png", dpi=500)


def save_history_images_mse(history_path_file, name, final_dest):
    #rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    #rc('text', usetex=True)

    path = history_path_file

    df = pd.read_excel(path)

    nrows = len(df['mean_squared_error'])

    acc = df['mean_squared_error'][0:nrows]
    val_acc = df['val_mean_squared_error'][0:nrows]

    plt.plot(acc, color="black")
    plt.plot(val_acc, linestyle='--', color="black")
    plt.title('model MSE')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid(True)
    plt.savefig(final_dest + "/" + name + "_mse.png", dpi=500)


def toexcel(history_callback, name):
    # os.chdir("/Users/Lorenzo/Desktop/")
    df = pd.DataFrame(history_callback.history)
    df.to_excel(name + ".xls")


def run(epoch, size, batch_size, data_path, results_path, dense_level, isCheckPoint, checkpoint_folder):
    id_result = str(uuid.uuid1())

    # results_path = "/home/loretto/Desktop/diabetic/results"

    num_classes = 5  # 0, 1, 2, 3, 4

    size = size
    target_size = (size, size)
    num_epochs = epoch

    # data_path = "/home/loretto/Desktop/dataset"

    train_dir = data_path + "/train"
    val_dir = data_path + "/val"

    # data generator (aug)
    rescale = 1. / 255
    train_datagen = ImageDataGenerator(
        rescale=rescale,
        featurewise_center=True,
        shear_range=0.2,
        zoom_range=0.5,
        width_shift_range=0.5,
        horizontal_flip=True,
        rotation_range=130,
        zca_whitening=True)

    # train_datagen = ImageDataGenerator(
    #     rescale=rescale)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',
        color_mode="rgb",
        shuffle=True)

    validation_datagen = ImageDataGenerator(
        rescale=rescale,
        featurewise_center=True,
        shear_range=0.2,
        zoom_range=0.5,
        width_shift_range=0.5,
        horizontal_flip=True,
        rotation_range=130,
        zca_whitening=True)

    # validation_datagen = ImageDataGenerator(
    #     rescale=rescale)

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',
        color_mode="rgb",
        shuffle=True)

    sample_size = target_size[0]

    if (isCheckPoint):
        try:
            model = load_model(checkpoint_folder + "/model.h5")
        except:

            print("[INFO] modello non presente nella cartella...creazione di uno nuovo")
            model = Sequential()

            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(target_size[0], target_size[0], 3),
                             padding='same'))
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            if (sample_size >= 64):
                model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
                model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            if (sample_size >= 128):
                model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
                model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            if (sample_size >= 256):
                model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
                model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            if (sample_size >= 512):
                model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
                model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            if (sample_size >= 1024):
                model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same'))
                model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            model.add(Flatten())
            model.add(Dense(dense_level, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1))
    else:
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(target_size[0], target_size[0], 3),
                         padding='same'))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        if (sample_size >= 64):
            model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        if (sample_size >= 128):
            model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        if (sample_size >= 256):
            model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        if (sample_size >= 512):
            model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        if (sample_size >= 1024):
            model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(dense_level, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))

    model.compile(loss='mse', optimizer='adadelta', metrics=['mse'])

    try:
        history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=train_generator.n // train_generator.batch_size,
                                      epochs=num_epochs,
                                      validation_data=validation_generator,
                                      validation_steps=validation_generator.n // validation_generator.batch_size)

    except Exception as e:
        print(
            "[ERROR] Possibile errore legato a file corrotti, o alla discrepanza tra la dimensione dei sample gestita dal ricaricato tramite checkpoint e la dimensione effettiva caricata nel corrente apprendimento")
        print(e)

    print("[INFO] Creazione cartella risultati...")

    final_dest = results_path + "/" + id_result
    if not os.path.exists(final_dest):
        os.makedirs(final_dest)

    try:
        print("[INFO] Salvataggio history...")
        df = pd.DataFrame(history.history)

        history_path = final_dest + "/" + "history.xls"
        df.to_excel(history_path)
    except Exception as e:
        print("[ERROR] Salvataggio training history")
        print(str(e))

    print("[INFO] Predizione su test set...")

    # Test Data
    rescale = 1. / 255
    test_dir = val_dir
    test_datagen = ImageDataGenerator(
        rescale=rescale,
        featurewise_center=True,
        shear_range=0.2,
        zoom_range=0.5,
        width_shift_range=0.5,
        horizontal_flip=True,
        rotation_range=130,
        zca_whitening=True)

    # test_datagen = ImageDataGenerator(
    #     rescale=rescale)

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

    for i in range(0, int(number_of_generator_calls)):
        test_labels.extend(np.array(test_generator[i][1]))

    test_labels = np.array(test_labels)

    pr = model.predict_generator(test_generator, steps=len(test_generator))


    # Arrotonda all'intero più vicino
    predicted = np.ceil(pr)

    #predicted[predicted >= 4] = 4

    true = test_labels

    print("[INFO] Salvataggio modello (checkpoint) in corso...")
    model.save(checkpoint_folder + "/model.h5")

    print("[INFO] Salvataggio risultati in corso...")
    print("[INFO] Calcolo matrice di confusione su test set...")

    cnf_matrix = confusion_matrix(true, predicted)

    print("[INFO] Calcolo coefficiente K di Cohen su test set...")

    c = cohen_kappa_score(predicted, true)

    print("[INFO] Calcolo accuracy score su test set...")

    acc = accuracy_score(true, predicted)

    print("[INFO] Salvataggio risultati...")

    parameters = {"cohen": c, "acc": acc}

    try:
        df = pd.DataFrame(pr)
        df.to_excel(final_dest + "/" + "test_predicted" + ".xls")
    except Exception as e:
        print("[ERROR] Salvataggio file excel predizioni")
        print(e)

    try:
        df = pd.DataFrame(true)
        df.to_excel(final_dest + "/" + "test_true" + ".xls")
    except Exception as e:
        print("[ERROR] Salvataggio file excel labels originali")
        print(e)

    try:
        df = pd.DataFrame(list(parameters.items()), columns=["cohen", "acc"])
        df.to_excel(final_dest + "/" + "parameters" + ".xls")
    except Exception as e:
        print("[ERROR] Salvataggio file parametri")
        print(e)

    try:
        df = pd.DataFrame(cnf_matrix)
        matrix_path = final_dest + "/" + "cm" + ".xls"
        df.to_excel(matrix_path)
    except Exception as e:
        print("[ERROR] Salvataggio file matrice di confusione")
        print(e)

    try:
        metadata = [target_size[0], num_epochs, batch_size]
        df = pd.DataFrame(metadata)
        df.to_excel(final_dest + "/" + "metadata" + ".xls")
    except Exception as e:
        print("[ERROR] Salvataggio file metadati pipeline apprendimento")
        print(e)

    try:
        print("[INFO] Creazione immagini history...")
        save_history_images_mse(history_path, id_result, final_dest)
    except Exception as e:
        print("[ERROR] Creazione immagini history")
        print(e)

    print("[INFO] Fine")

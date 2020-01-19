from __future__ import division

import numpy as np
import pandas as pd
import uuid
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from keras.utils import plot_model

from keras.applications import VGG16

def toexcel(history_callback, name):
    df = pd.DataFrame(history_callback.history)
    df.to_excel(name + ".xls")


def run(epoch, size, batch_size, data_path, results_path, dense_level):
    id = str(uuid.uuid1())

    color_modes = ["grayscale", "rgb"]

    num_classes = 5  # 0, 1, 2, 3, 4

    image_size = size

    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

    target_size = (image_size, image_size)
    num_epochs = epoch
    color_mode = color_modes[1]

    train_dir = data_path + "/train"
    val_dir = data_path + "/val"

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

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        class_mode='categorical',
        color_mode=color_mode,
        batch_size=batch_size,
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

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=True)

    channels = 3

    if color_mode == "grayscale":
        channels = 1

    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    for layer in vgg_conv.layers:
        print(layer, layer.trainable)

    model = Sequential()
    model.add(vgg_conv)
    model.add(Flatten())
    model.add(Dense(dense_level, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_generator.n // train_generator.batch_size,
                                  epochs=num_epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_generator.n // validation_generator.batch_size)

    final_dest = results_path + "/" + id

    if not os.path.exists(final_dest):
        os.makedirs(final_dest)

    #plot_model(model, to_file=final_dest + "/" + "model.png", show_shapes=True, show_layer_names=True)

    df = pd.DataFrame(history.history)
    df.to_excel(final_dest + "/" + "history.xls")

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

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        class_mode='categorical',
        batch_size=batch_size,
        color_mode=color_mode,
        shuffle=True)

    pr = model.predict_generator(test_generator, steps=len(test_generator))

    predicted = []

    for p in pr:
        predicted.append(p)

    predicted = np.array(predicted)

    true = []

    for l in test_generator.labels:
        true.append(l)

    true = np.array(true)

    cnf_matrix = confusion_matrix(true, predicted)

    c = cohen_kappa_score(predicted, true)

    acc = accuracy_score(true, predicted)

    parameters = {}

    parameters["cohen"] = c
    parameters["acc"] = acc

    df = pd.DataFrame(predicted)
    df.to_excel(final_dest + "/" + "test_predicted" + ".xls")

    df = pd.DataFrame(true)
    df.to_excel(final_dest + "/" + "test_true" + ".xls")

    df = pd.DataFrame(list(parameters.items()), columns=["cohen", "acc"])
    df.to_excel(final_dest + "/" + "parameters" + ".xls")

    df = pd.DataFrame(cnf_matrix)
    df.to_excel(final_dest + "/" + "cm" + ".xls")

    metadata = [target_size[0], num_epochs, batch_size]

    df = pd.DataFrame(metadata)
    df.to_excel(final_dest + "/" + "metadata" + ".xls")

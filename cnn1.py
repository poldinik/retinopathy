from __future__ import division

import numpy as np
import pandas as pd
import uuid
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from keras.utils import plot_model


def toexcel(history_callback, name):
    # os.chdir("/Users/Lorenzo/Desktop/")
    df = pd.DataFrame(history_callback.history)
    df.to_excel(name + ".xls")


def run(epoch, size, batch_size, data_path, results_path, dense_level):
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
    # train_datagen = ImageDataGenerator(
    #     rescale=rescale,
    #     featurewise_center=True,
    #     shear_range=0.2,
    #     zoom_range=0.5,
    #     width_shift_range=0.5,
    #     horizontal_flip=True,
    #     rotation_range=130,
    #     zca_whitening=True)

    train_datagen = ImageDataGenerator(
        rescale=rescale)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode="rgb",
        shuffle=True)

    # validation_datagen = ImageDataGenerator(
    #     rescale=rescale,
    #     featurewise_center=True,
    #     shear_range=0.2,
    #     zoom_range=0.5,
    #     width_shift_range=0.5,
    #     horizontal_flip=True,
    #     rotation_range=130,
    #     zca_whitening=True)

    validation_datagen = ImageDataGenerator(
        rescale=rescale)

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode="rgb",
        shuffle=True)

    sample_size = target_size[0]

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
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_generator.n // train_generator.batch_size,
                                  epochs=num_epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_generator.n // validation_generator.batch_size)

    final_dest = results_path + "/" + id_result

    if not os.path.exists(final_dest):
        os.makedirs(final_dest)

    #plot_model(model, to_file=final_dest + "/" + "model.png", show_shapes=True, show_layer_names=True)

    df = pd.DataFrame(history.history)
    df.to_excel(final_dest + "/" + "history.xls")

    # Test Data
    rescale = 1. / 255
    test_dir = val_dir
    # test_datagen = ImageDataGenerator(
    #     rescale=rescale,
    #     featurewise_center=True,
    #     shear_range=0.2,
    #     zoom_range=0.5,
    #     width_shift_range=0.5,
    #     horizontal_flip=True,
    #     rotation_range=130,
    #     zca_whitening=True)

    test_datagen = ImageDataGenerator(
        rescale=rescale)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        class_mode='categorical',
        batch_size=batch_size,
        color_mode="rgb",
        shuffle=True)

    pr = model.predict_generator(test_generator, steps=len(test_generator))

    predicted = []

    # arg max poichè pr produce un array di valori sui quale trovare il massimo
    for p in pr:
        predicted.append(np.argmax(p))

    predicted = np.array(predicted)

    true = []

    for l in test_generator.labels:
        true.append(l)

    true = np.array(true)

    try:
        cnf_matrix = confusion_matrix(true, predicted)
        df = pd.DataFrame(cnf_matrix)
        df.to_excel(final_dest + "/" + "cm" + ".xls")
    except:
        pass


    try:
        c = cohen_kappa_score(predicted, true)
    except:
        pass

    try:
        acc = accuracy_score(true, predicted)
    except:
        pass


    try:
        parameters = {"cohen": c, "acc": acc}
        df = pd.DataFrame(list(parameters.items()), columns=["cohen", "acc"])
        df.to_excel(final_dest + "/" + "parameters" + ".xls")
    except:
        pass


    df = pd.DataFrame(predicted)
    df.to_excel(final_dest + "/" + "test_predicted" + ".xls")

    df = pd.DataFrame(true)
    df.to_excel(final_dest + "/" + "test_true" + ".xls")


    metadata = [target_size[0], num_epochs, batch_size]

    df = pd.DataFrame(metadata)
    df.to_excel(final_dest + "/" + "metadata" + ".xls")

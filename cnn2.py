from __future__ import division

import numpy as np
import pandas as pd
import uuid
import random
import os
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from keras.utils import plot_model

from keras.applications import VGG16, VGG19, Xception

# Load Dataset

def toexcel(history_callback, name):
    #os.chdir("/Users/Lorenzo/Desktop/")
    df = pd.DataFrame(history_callback.history)
    df.to_excel(name + ".xls")

id = str(uuid.uuid1())

results_path = "/home/loretto/Desktop/diabetic/results"
color_modes = ["grayscale", "rgb"]

num_classes = 5  # 0, 1, 2, 3, 4

image_size = 256


#Load the VGG model
#vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
vgg_conv = VGG19(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

#vgg_conv = Xception(include_top=True, weights='imagenet', input_shape=(image_size, image_size, 3),classes=1000)

target_size = (image_size, image_size)
num_epochs = 10
color_mode = color_modes[1]

data_path = "/home/loretto/Desktop/dataset"
train_dir = data_path + "/train"
val_dir = data_path + "/val"

# data generator (aug)
rescale = 1. / 255
train_datagen = ImageDataGenerator(
    rescale=rescale,
    featurewise_center=True,
    rotation_range=130,
    zca_whitening=True)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    class_mode='categorical',
    color_mode=color_mode,
    shuffle=True)

validation_datagen = ImageDataGenerator(
    rescale=rescale,
    featurewise_center=True,
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

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

model = Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(size, size, channels)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(target_size[0], target_size[0], 3), padding='same'))
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
# model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
# model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


# 1st Conv layer
# model.add(Conv2D(16, kernel_size=(13, 13), activation='relu', input_shape=(target_size[0], target_size[0], 3), padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# # 2nd Conv layer
# model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'))
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# # 3rd Conv layer
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# # 4th Conv layer
# model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
# # 5th Conv layer
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Fully-Connected layer
#model.add(Flatten())
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(num_classes, activation='softmax'))

# model.summary()

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# Train Model
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=train_generator.n // train_generator.batch_size,
                              epochs=num_epochs,
                              validation_data=validation_generator,
                              validation_steps=validation_generator.n // validation_generator.batch_size)


final_dest = results_path + "/" + id

if not os.path.exists(final_dest):
    os.makedirs(final_dest)

plot_model(model, to_file=final_dest + "/" + "model.png", show_shapes=True, show_layer_names=True)

# accuracy = history.history['accuracy']
# val_accuracy = history.history['val_accuracy']
# plt.plot(range(len(accuracy)), accuracy, color='blue', label='Training accuracy')
# plt.plot(range(len(accuracy)), val_accuracy, color='red', label='Validation accuracy')
# plt.xlabel('Epoch No.')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()


toexcel(history, id + "_history-vgg19")

df = pd.DataFrame(history.history)
df.to_excel(final_dest + "/" + "history-vgg19.xls")

# Test Data Generator
rescale = 1. / 255
test_dir = val_dir
test_datagen = ImageDataGenerator(
    rescale=rescale,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    class_mode='categorical',
    batch_size=32,
    color_mode=color_mode,
    shuffle=True)

# result = model.evaluate_generator(test_generator, steps=len(test_generator))

# df = pd.DataFrame(result)
# df.to_excel(final_dest + "/" + "test" + ".xls")

pr = model.predict_generator(test_generator, steps=len(test_generator))

print(pr)
print(pr.shape)

print(test_generator.labels)
print(test_generator.labels.shape)

predicted = []

for p in pr:
    predicted.append(np.argmax(p))

print(predicted)
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
# parameters["f1"] = f1
parameters["acc"] = acc


df = pd.DataFrame(predicted)
df.to_excel(final_dest + "/" + "test_predicted" + ".xls")

df = pd.DataFrame(true)
df.to_excel(final_dest + "/" + "test_true" + ".xls")

df = pd.DataFrame(list(parameters.items()), columns=["cohen", "acc"])
df.to_excel(final_dest + "/" + "parameters" + ".xls")

df = pd.DataFrame(cnf_matrix)
df.to_excel(final_dest + "/" + "cm" + ".xls")

metadata = []
metadata.append(target_size[0])
metadata.append(num_epochs)

df = pd.DataFrame(metadata)
df.to_excel(final_dest + "/" + "metadata" + ".xls")

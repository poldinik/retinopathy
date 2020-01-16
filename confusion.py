import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
import os
import argparse
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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


#
# cnf_matrix = confusion_matrix(y_test, preticted)
#
# class_names = ["0", "1", "2", "3", "4"]
#
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
#                           title='Confusion matrix' ' fold\n' + str(
#                               round(acc, 2) * 100) + '% accuracy' + ' Cohen K: ' + str(round(c, 2)))
# plt.savefig(str(i) + "fold" + ".png")


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the xls")
ap.add_argument("-n", "--name", required=True, help="name of files output")
ap.add_argument("-t", "--title", required=True, help="title of figure")
args = vars(ap.parse_args())

path = args["path"]
name = args["name"]
title = args["title"]

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

plt.savefig(name + "_cm" + ".png", dpi=500)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import argparse
import numpy as np
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the xls")
ap.add_argument("-n", "--name", required=True, help="name of files output")
args = vars(ap.parse_args())

path = args["path"]
name = args["name"]

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
plt.savefig(name + "_acc.png", dpi=500)


plt.show()
# summarize history for loss
plt.plot(loss, color="black")
plt.plot(val_loss, linestyle='--', color="black")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.savefig(name + "_loss.png", dpi=500)
plt.show()


import numpy as np
import cv2
import os



def load_fundus_images(df, inputPath, size):
    images = []

    for i in df.index.values:

        try:
            name = df["image"][i]
            basePath = os.path.sep.join([inputPath, name + ".jpeg"])
            # print(basePath)

            image = cv2.imread(basePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (int(size), int(size)))
            images.append(image)
        except:
            pass
    return np.array(images)

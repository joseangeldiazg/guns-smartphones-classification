# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from natsort import natsorted, ns
from imutils import paths
import random
import numpy as np
import argparse
import imutils
import os
import csv
import cv2
import pandas as pd

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model model")
ap.add_argument("-t", "--testpath", required=True, help="path to input images")
ap.add_argument("-i", "--image", required=False, help="path to input image")
args = vars(ap.parse_args())

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

images = natsorted(list(paths.list_images(args["testpath"])),alg=ns.IGNORECASE)
random.seed(42)

with open("./Output/salida.csv","w", newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["ID"]+["Ground_Truth"])
    for imagepath in images:
        image = cv2.imread(imagepath)
        image = cv2.resize(image, (64, 64))
        orig = image.copy()
        # pre-process the image for classification
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        # classify the input image
        (pistol, smartphone) = model.predict(image)[0]
        # build the label
        label = "0" if pistol > smartphone else "1"
        spamwriter.writerow([imagepath.replace("./Test/","",1)]+[label])
        proba = pistol if pistol > smartphone else smartphone
        label = "{}: {:.2f}%".format(label, proba * 100)

#Mostrar imagen:

#output = imutils.resize(orig, width=400)
#cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
#	0.7, (0, 255, 0), 2)
# show the output image
#cv2.imshow("Output", output)
#cv2.waitKey(0)

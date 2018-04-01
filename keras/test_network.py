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
ap.add_argument("-t", "--testpath", required=True, help="path to input test images")
args = vars(ap.parse_args())

save_dir = "./files"

#Cargamos los mejores pesos del entrenamiento y guardamos el modelo
print("[INFO] serializing network...")
model = load_model("./files/checkpoints/weights-improvement.hdf5")

#model.evaluate(X_test, Y_test, verbose=1) # Evaluate the trained model on the test set!

images = natsorted(list(paths.list_images(args["testpath"])),alg=ns.IGNORECASE)
random.seed(42)

output_dir = os.path.join(save_dir,"output")
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
with open("./files/output/salida1_7_incial_64filter_da_532_64filter.csv","w", newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["ID"]+["Ground_Truth"])
    for imagepath in images:
        image = cv2.imread(imagepath)
        image = cv2.resize(image, (64, 64))
        # pre-process the image for classification
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        image = np.array(image, dtype="float") / 255.0
        
        # classify the input image
        (pistol, smartphone) = model.predict(image)[0]
        # build the label
        label = "0" if pistol > smartphone else "1"
        spamwriter.writerow([imagepath.replace("./Test/","",1)]+[label])
        proba = pistol if pistol > smartphone else smartphone
        label = "{}: {:.2f}%".format(label, proba * 100)
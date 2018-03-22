# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from pyimagesearch.lenet import LeNet

EPOCHS = 1000
INIT_LR = 1e-3
BS = 32

model = LeNet.build(width=64, height=64, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

model.load_weights("./Models/checkpoints/bestmodel.hdf5")
model.save("./Models/1000ep.model")

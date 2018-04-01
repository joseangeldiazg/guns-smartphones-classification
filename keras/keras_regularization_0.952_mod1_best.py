from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Dense, Flatten, Convolution2D, MaxPooling2D, Dropout, average
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.regularizers import l2 # L2-regularisation
from keras.layers.normalization import BatchNormalization # batch normalisation
from keras.preprocessing.image import ImageDataGenerator # data augmentation
from keras.callbacks import EarlyStopping # early stopping
# import the necessary packages
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from natsort import natsorted, ns
from keras.callbacks import ModelCheckpoint
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import csv
import random
import cv2,os
import pandas as pd

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input train images")
args = vars(ap.parse_args())

batch_size = 64 # in each iteration, we consider 140 training examples at once
num_epochs = 20 # we iterate at most fifty times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # use 32 kernels in both convolutional layers
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 128 # there will be 128 neurons in both hidden layers
l2_lambda = 0.0001 # use 0.0001 as a L2-regularisation factor
ens_models = 3 # we will train three separate models on the data
save_dir = "./files"

#num_train = 60000 # there are 60000 training examples in MNIST
#num_test = 10000 # there are 10000 test examples in MNIST

height, width, depth = 64, 64, 3 
num_classes = 2 # there are 10 classes (1 per digit)

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)
# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (height, width))
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "./Train/Smartphone" else 0
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 20% for testing
(X_train, X_test, Y_train, Y_test) = train_test_split(data,labels, test_size=0.2, random_state=42)
#(X_train, y_train), (X_test, y_test) = mnist.load_data() # fetch MNIST data

#X_train = X_train.reshape(X_train.shape[0], height, width, depth)
#X_test = X_test.reshape(X_test.shape[0], height, width, depth)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255 # Normalise data to [0, 1] range
X_test /= 255 # Normalise data to [0, 1] range

Y_train = np_utils.to_categorical(Y_train, num_classes) # One-hot encode the labels
Y_test = np_utils.to_categorical(Y_test, num_classes) # One-hot encode the labels
# Explicitly split the training and validation sets
X_val = X_train[532:]
Y_val = Y_train[532:]
X_train = X_train[:532]
Y_train = Y_train[:532]

inp = Input(shape=(height, width, depth)) # N.B. TensorFlow back-end expects channel dimension last
inp_norm = BatchNormalization()(inp) # Apply BN to the input (N.B. need to rename here)

outs = [] # the list of ensemble outputs
for i in range(ens_models):
    # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer), applying BN in between
    conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), 
        padding='same', 
        kernel_initializer='he_uniform', 
        kernel_regularizer=l2(l2_lambda), 
        activation='relu')(inp_norm)
    conv_1 = BatchNormalization()(conv_1)
    
    conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), 
        padding='same', 
        kernel_initializer='he_uniform', 
        kernel_regularizer=l2(l2_lambda), 
        activation='relu')(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
    drop_1 = Dropout(drop_prob_1)(pool_1)
    flat = Flatten()(drop_1)
    hidden = Dense(hidden_size, 
        kernel_initializer='he_uniform', 
        kernel_regularizer=l2(l2_lambda), 
        activation='relu')(flat) # Hidden ReLU layer
    hidden = BatchNormalization()(hidden)
    drop = Dropout(drop_prob_2)(hidden)
    outs.append(Dense(num_classes, 
        kernel_initializer='glorot_uniform', 
        kernel_regularizer=l2(l2_lambda), 
        activation='softmax')(drop)) # Output softmax layer

out = average(outs) # average the predictions to obtain the final output

model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

datagen = ImageDataGenerator(
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)
datagen.fit(X_train)

#Checkpoint
ckpt_dir = os.path.join(save_dir,"checkpoints")
if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)

filepath = os.path.join(ckpt_dir,"weights-improvement.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
print("Saving improvement checkpoints to \n\t{0}".format(filepath))

# early stop callback, given a bit more leeway
stahp = EarlyStopping(monitor="val_loss", patience=10)
# fit the model on the batches generated by datagen.flow() - most parameters similar to model.fit
model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0],
                        epochs=num_epochs,
                        validation_data=(X_val, Y_val),
                        verbose=1,
                        callbacks=[checkpoint,stahp]) # adding early stopping

#Cargamos los mejores pesos del entrenamiento y guardamos el modelo
print("[INFO] serializing network...")
model.load_weights("./files/checkpoints/weights-improvement.hdf5")
model.save("./files/checkpoints/weights-improvement.hdf5")
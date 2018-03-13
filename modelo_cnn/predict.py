import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import csv

def load_test(test_path,image_size):
    images = []
    #labels = []
    img_names = []
    #cls = []

    print('Going to read test images')
    
    path = os.path.join(test_path, '*g')
    files = glob.glob(path)
    for fl in files:
        image = cv2.imread(fl)
        image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        images.append(image)
        #label = np.zeros(len(classes))
        #label[index] = 1.0
        #labels.append(label)
        flbase = os.path.basename(fl)
        img_names.append(flbase)
        #cls.append(fields)
    images = np.array(images)
    #labels = np.array(labels)
    img_names = np.array(img_names)
    #cls = np.array(cls)

    return images, names


# First, pass the path of the image
#dir_path = os.path.dirname(os.path.realpath(__file__))
#image_path=sys.argv[1] 
filename =  "D:/Manuel/Documents/Master/MineriaDatosAvanzados/SihamTabik.github.io-master/Competicion/Test/"
image_size=128
num_channels=3
images = []
names = []
images, names = load_test(filename,image_size)

print(names)
clase = []
contador = 0
for image in images:
	# Reading the image using OpenCV
	#image = cv2.imread(filename)
	# Resizing the image to our desired size and preprocessing will be done exactly as done during training
	#print(filename)
	#image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
	#images.append(image)
	#images = np.array(images, dtype=np.uint8)
	#images = images.astype('float32')
	#images = np.multiply(images, 1.0/255.0) 
	#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
	x_batch = image.reshape(1, image_size,image_size,num_channels)

	## Let us restore the saved model 
	sess = tf.Session()
	# Step-1: Recreate the network graph. At this step only graph is created.
	saver = tf.train.import_meta_graph('./result/modelo1.meta')
	# Step-2: Now let's load the weights saved using the restore method.
	saver.restore(sess, tf.train.latest_checkpoint('./result/'))

	# Accessing the default graph which we have restored
	graph = tf.get_default_graph()

	# Now, let's get hold of the op that we can be processed to get the output.
	# In the original network y_pred is the tensor that is the prediction of the network
	y_pred = graph.get_tensor_by_name("y_pred:0")

	## Let's feed the images to the input placeholders
	x= graph.get_tensor_by_name("x:0") 
	y_true = graph.get_tensor_by_name("y_true:0") 
	y_test_images = np.zeros((1, 2)) 


	### Creating the feed_dict that is required to be fed to calculate y_pred 
	feed_dict_testing = {x: x_batch, y_true: y_test_images}
	result=sess.run(y_pred, feed_dict=feed_dict_testing)
	# result is of this format [probabiliy_of_rose probability_of_sunflower]
	print(result)
	if result[0][0] > result[0][1]:
		clase[names[contador]] = 0
	else:
		clase[names[contador]] = 1
	contador +=1

print(clase)
#with open("./result/sample1.csv", "wb") as csv_file:
#	writer = csv.writer(csv_file, delimiter=',')
#	for line in clase:


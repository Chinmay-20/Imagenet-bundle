#python orient_images.py --db indoor_cvpr/hdf5/orientation_features.hdf5 --dataset indoor_cvpr/rotated_images --model models/orientation.cpickle

from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import progressbar
import pickle
import imutils
import h5py
import cv2
import argparse
import random
import os

ap=argprae.ArgumentParser()
ap.add_argument("-d","--db",required=True,help="path HDF5 database")
ap.add_argument("-i","--dataset",required=True,help="path to input images dataset")
ap.add_argument("-m","--model",required=True,help="path to trained orientation model")
args=vars(ap.parse_args())

db=h5py.File(args["db"])
labelNames=[int(angle) for angle in db["label_names"][:]]
db.close()

print("[INFO] sampling images...")
imagePaths=list(paths.list_images(args["dataset"]))
imagePaths=np.random.choice(imagePaths,size=(10,),replace=True)

print("[INFO] loading network...")
vgg=VGG16(weights="imagenet",include_top=False)
print("[INFO] loading model...")
model=pickle.loads(open(args["model"],"rb").read())

for imagePath in imagePaths:
	orig=cv2.imread(imagePath)
	
	image=load_img(imagePath,target_size=(224,224))
	image=img_to_array(image)
	
	image=np.expand_dims(image,axis=0)
	image=imagenet_utils.preprocess_input(image)
	
	features=vgg.predict(image)
	features=features.reshape((features.shape[0],512*7*7))
	
	angle=model.predict(features)
	angle=labelNames[angle[0]]
	
	rotated=imutils.rotate_bound(orig,360-angle)
	cv2.imshow("Original",orig)
	cv2.imshow("Corrected",rotated)
	cv2.waitKey(0)
	
	

#python create_dataset.py --dataset indoor_cvpr/images --output indoor_cvpr/rotated_images

from imutils import paths
import numpy as np
import progressbar
import argparse
import imutils
import random
import cv2
import os

ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input directory of images")
ap.add_argument("-o","--output",required=True,help="path to output directory of rotated images")
args=vars(ap.parse_args())

imagePaths=list(paths.list_images(args["dataset"]))[:10000]
random.shuffle(imagePaths)

angles={}
widgets=["Building Dataset: ",progressbar.Percentage()," ",progressbar.Bar()," ",progressbar.ETA()]
pbar=progressbar.ProgressBar(maxval=len(imagePaths),widgets=widgets).start()

for (i,imagePath) in enumerate(imagePaths):
	angle=np.random.choice([0,90,180,270])
	image=cv2.imread(imagePath)
	
	if image is None:
		continue
		
	image=imutils.rotate_bound(image,angle)
	base=os.path.sep.join([args["output"],str(angle)])
	
	ext=imagePath[imagePath.rfind(".")]
	outputPath=[base,"image{}_{}".format(str(angles.get(angle,0)).zfill(5),ext)]
	outputPath=os.path.sep.join(outputPath)
	
	cv2.imwrite(outputPath,image)
	
	c=angles.get(angle,0)
	angles[angle]=c+1
	pbar.update(i)

pbar.finish()

for angle in sorted(angle.keys()):
	print("[INFO] angle={}: {:,}".format(angle,angles[angle]))
	
	

#python build_dataset.py

#~/mxnet/bin/im2rec /raid/datasets/cars/lists/train.lst "" /raid/datasets/cars/rec/train.rec resize=256 encoding=’.jpg’ quality=100
#~/mxnet/bin/im2rec /raid/datasets/cars/lists/test.lst "" /raid/datasets/cars/rec/test.rec resize=256 encoding=’.jpg’ quality=100
#~/mxnet/bin/im2rec /raid/datasets/cars/lists/val.lst "" /raid/datasets/cars/rec/val.rec resize=256 encoding=’.jpg’ quality=100
import car_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import progressbar
import pickle
import os


print("[INFO] loading image paths and labels...")
rows=open(config.LABELS_PATH).read()
rows=rows.strip().split("\n")[1:]
trainPaths=[]
trainLabels=[]

for row in rows:
	(filename,make,model)=row.split(",")[:3]
	filename=filename[filename.rfind("/")+1:]
	trainPaths.append(os.sep.join([config.IMAGES_PATH,filename]))
	trainLabels.append("{}:{}".format(make,model))
	
	
numVal=int(len(trainPaths)*config.NUM_VAL_IMAGES)
numTest=int(len(trainPaths)*config.NUM_TEST_IMAGES)

print("[INFO] ecoding labels...")
lee=LabelEncoder().fit(trainLabels)
trainLabels=le.transform(trainLabels)


print("[INFO] constructing validation data...")
split=train_test_split(trainPaths,trainLabels,test_size=numVal,stratify=trainLabels)
(trainPaths,valPaths,trainLabels,valLabels)=split


print("[INFO] constructing test data...")
split=train_test_split(trainPaths,trainLabels,test_size=numTest,stratify=trainLabels)
(trainPaths,testPaths,trainLabels,testLabels)=split

datasets=[
	("train",trainPaths,trainLabels,config.TRAIN_MX_LIST),
	("val",valPaths,valLabels,config.VAL_MX_LIST),
	("test",testPaths,testLabels,config.TEST_MX_LIST)]
	
for (dType,paths,labels,outputPath) in datasets:
	print("[INFO] building {}...".format(outputPath))
	f=open(outputPath,"w")
	
	widgets=["Building List: ",progressbar.Percentage()," ",progresbar.Bar()," ",progressbar.ETA()]
	pbar=progressbar.ProgressBar(maxval=len(paths),widgets=widgets).start()
	
	for (i,(path,label)) in enumerate(zip(paths,labels)):
		row="\t".join([str(i),str(label),path])
		f.write("{}\n".format(row))
		pbar.update(i)
		
	pbar.finish()
	f.close()

print("[INFO] serializing label encoder...")
f=open(config.LABEL_ENCODER_PATH,"wb")
f.write(pickle.dumps(le))
f.close()



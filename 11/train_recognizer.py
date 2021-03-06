#python train_recognizer.py --checkpoints checkpoints

#python train_recognizer.py --checkpoints checkpoints
#python train_recognizer.py --checkpoints checkpoints --model checkpoints/epoch_30.hdf5 --start-epoch 30

#python train_recognizer.py --checkpoints checkpoints
#python train_recognizer.py --checkpoints checkpoints --model checkpoints/epoch_40.hdf5 --start-epoch 40
#python train_recognizer.py --checkpoints checkpoints --model checkpoints/epoch_60.hdf5 --start-epoch 60
import matplotlib
matplotlib.use("Agg")

import emotion_config as config
from imagetoarraypreprocessor import ImagetoArrayPreprocessor
from epochcheckpoint import EpochCheckpoint
from trainingmonitor import TrainingMonitor
from hdf5datasetgenerator import HDF5DatasetGenerator
from emotionvggnet import EmotionVGGNet
from keras.preprocessing.image import ImageDataGenerator4
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K

ap=argparse.ArgumentParser()
ap.add_argument("-c","--checkpoints",required=True,help="path to output checkpoint directory")
ap.add_argument("-m","--model",type=str,help="path to *specific* checkpoint to load")
ap.add_argument("-s","--start-epoch",type=int,default=0,help="epoch to restart training at")
args=vars(ap.parse_args())

trainAug=ImageDataGenerator(rotation_range=10,zoom_range=0.1,horizontal_flip=True,rescale=1/255.0.fill_model="nearest")
valAug=ImageDataGenerator(rescale=1/255.0)
iap=ImagetoArrayPreprocesssor()

trainGen=HDF5DatasetGenerator(config.TRAIN_HDF5,config.BATCH_SIZE,aug=trainAug,preprocessors=[iap],classes=config.NUM_CLASSES)
valGen=HDF5DatasetGenerator(config.VAL_HDF5,config.BATCH_SIZE,aug=valAug,preprocessors=[iap],classes=config.NUM_CLASSES)

if args["model"] is None:
	print("[INFO] compiling model...")
	model=EmotionVGGNet.build(width=48,height=48,depth=1,classes=config.NUM_CLASSES)
	opt=Adam(lr=1e-3)
	model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
else:
	print("[INFO] loading {}...".format(args["model"]))
	model=load_model(args["model"])
	
	print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr,1e-3)
	print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))
	
figPath=os.path.sep.join([config.OUTPUT_PATH,"vggnet_emotion.png"])
jsonPath=os.path.sep.join([config.OUTPUT_PATH,"vggnet_emotion.json"])
callbacks=[
	EpochCheckpoint(args["checkpoints"],every=5,startAt=args["start_epoch"]),
	TrainingMonitor(figPath,jsonPath=jsonPath,startAt=args["start_epoch"])]
	
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages//config.BATCH_SIZE,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages//config.BATCH_SIZE,
	epochs=15,
	max_queue_size=config.BATCH_SIZE*2,
	callbacks=callbacks,verbose=1)
	
trainGen.close()
valGen.close()



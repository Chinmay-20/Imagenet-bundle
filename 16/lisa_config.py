import os

BASE_PATH="lisa"

ANNOT_PATH=os.path.sep.join([BASE_PATH,"allAnnotations.csv"])

TRAIN_RECORD=os.path.sep.join([BASE_PATH,"records/training.record"])

TEST_RECORD=os.path.sep.join([BASE_PATH,"records/classes.pbtxt"])

TEST_SIZE=0.25

CLASSES={"pedestrianCrossing":1,"siggnalAhead":2,"stop":3}

from keras.models import load_model
from hdf5.hdf5datasetgenerator import HDF5DatasetGenerator
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.meanpreprocessor import MeanPreprocessor
from preprocessing.croppreprocessor import CropPreprocessor

import config.config as config
import json
from utils.roc_curve import plotRocCurve

import numpy as np



means = json.loads(open(config.DATASET_MEAN).read())



sp = SimplePreprocessor(config.IMAGE_SIZE, config.IMAGE_SIZE)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
cp = CropPreprocessor(config.IMAGE_SIZE, config.IMAGE_SIZE)
iap = ImageToArrayPreprocessor()



print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)


print("[INFO] predicting on test data (no crops)...")
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 5000, preprocessors=[sp, mp, iap], classes=3)

quantityImages = testGen.numImages // testGen.batchSize

predData = np.zeros(shape=(5000, config.NUM_CLASSES))
labelsData = np.zeros(shape=(5000, config.NUM_CLASSES))

for (i, (images, labels)) in enumerate(testGen.generator(passes=1)):

    pred = model.predict(images)

    predData = pred
    labelsData = labels


plotRocCurve(config.NUM_CLASSES, labelsData, predData)

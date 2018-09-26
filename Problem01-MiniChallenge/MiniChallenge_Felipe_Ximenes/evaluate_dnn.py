

# import the necessary packages
import config.config as config
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.meanpreprocessor import MeanPreprocessor
from preprocessing.croppreprocessor import CropPreprocessor
from hdf5.hdf5datasetgenerator import HDF5DatasetGenerator
from utils.rank import rank5_accuracy
from keras.models import load_model
import numpy as np
import progressbar
import json


means = json.loads(open(config.DATASET_MEAN).read())


sp = SimplePreprocessor(config.IMAGE_SIZE, config.IMAGE_SIZE)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
cp = CropPreprocessor(config.IMAGE_SIZE, config.IMAGE_SIZE)
iap = ImageToArrayPreprocessor()


print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)


print("[INFO] predicting on test data...")

testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATH_SIZE, preprocessors=[mp], classes=3)
predictions = []

widgets = ["Evaluating: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=testGen.numImages // config.BATH_SIZE, widgets=widgets).start()


for (i, (images, labels)) in enumerate(testGen.generator(passes=1)):

	for image in images:

		crops = cp.preprocess(image)
		crops = np.array([iap.preprocess(c) for c in crops],
			dtype="float32")


		pred = model.predict(crops)
		predictions.append(pred.mean(axis=0))


	pbar.update(i)


pbar.finish()
(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()
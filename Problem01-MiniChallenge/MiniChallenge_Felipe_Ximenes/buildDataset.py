# import the necessary packages
from config import config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from hdf5.hdf5datasetwriter import HDF5DatasetWriter

import pandas as pd
import numpy as np
import progressbar
import json
import cv2


df = pd.read_csv(config.CSVPATH, delimiter="\t")

dataPath = list(config.IMAGES_PATH + df['filename'])
labels = list(df['category'])

le = LabelEncoder()
trainLabels = le.fit_transform(labels)


split = train_test_split(dataPath, trainLabels, test_size=0.25, train_size=0.75, stratify=trainLabels, random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split


split = train_test_split(trainPaths, trainLabels, test_size=0.25, train_size=0.75, stratify=trainLabels, random_state=42)
(trainPaths, valPaths, trainLabels, valLabels) = split


datasets = [
	("train", trainPaths, trainLabels, config.TRAIN_HDF5),
	("val", valPaths, valLabels, config.VAL_HDF5),
	("test", testPaths, testLabels, config.TEST_HDF5)]


aap = AspectAwarePreprocessor(config.IMAGE_SIZE, config.IMAGE_SIZE)
(R, G, B) = ([], [], [])


for (dType, paths, labels, outputPath) in datasets:


	print("[INFO] building {}...".format(outputPath))
	writer = HDF5DatasetWriter((len(paths), config.IMAGE_SIZE, config.IMAGE_SIZE, 3), outputPath)


	widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
	pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()


	for (i, (path, label)) in enumerate(zip(paths, labels)):

		image = cv2.imread(path)
		image = aap.preprocess(image)


		if dType == "train":
			(b, g, r) = cv2.mean(image)[:3]
			R.append(r)
			G.append(g)
			B.append(b)


		writer.add([image], [label])
		pbar.update(i)


	pbar.finish()
	writer.close()


print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()
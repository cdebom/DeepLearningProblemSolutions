# import the necessary packages
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import config.config as config
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.patchpreprocessor import PatchPreprocessor
from preprocessing.meanpreprocessor import MeanPreprocessor
from training.monitorTraining import TrainingMonitor
from hdf5.hdf5datasetgenerator import HDF5DatasetGenerator
from dnn.resnet import ResNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os

INIT_LR = 1e-3

def poly_decay(epoch):

	NUM_EPOCHS = 100

	maxEpochs = NUM_EPOCHS
	baseLR = INIT_LR
	power = 1.0

	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

	return alpha


aug = ImageDataGenerator(rotation_range=20, zoom_range=0.10,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True,vertical_flip=True, fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(config.IMAGE_SIZE, config.IMAGE_SIZE)
pp = PatchPreprocessor(config.IMAGE_SIZE, config.IMAGE_SIZE)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()


trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATH_SIZE, aug=aug, preprocessors=[pp, mp, iap])
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATH_SIZE, preprocessors=[sp, mp, iap])


print("[INFO] compiling model...")

opt = Adam(lr=INIT_LR)
model = ResNet.build(config.IMAGE_SIZE,config.IMAGE_SIZE,3, 3, (9,9,9), (64,64,128,256), reg=0.0003)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(path)]


model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages // config.BATH_SIZE,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages // config.BATH_SIZE,
	epochs=100,
	max_queue_size=10,
	callbacks=callbacks, verbose=1)


print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

trainGen.close()
valGen.close()
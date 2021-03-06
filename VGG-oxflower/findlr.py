from __future__ import print_function

import numpy as np
import keras
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from VGG16 import VGG16Net
from clr import LRFinder
#import matplotlib.pyplot as plt
#Load oxflower17 dataset
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
x, y = oxflower17.load_data(one_hot=True)

#Split train and test data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2,shuffle = True)

#Call VGG16Net model
# input image dimensions
data_aug = True
img_rows, img_cols = 224, 224
# The CIFAR10 images are RGB.
img_channels = 3
nb_classes = 17
n_epochs = 1
n_batch = 8

num_sample = X_train.shape[0]

lrf = LRFinder(num_sample,
        n_batch,
        minimum_lr=1e-4,
        maximum_lr=1,
        lr_scale='exp',
        #validation_data = (X_test, Y_test),
        validation_sample_rate=1)

VGG16_model = VGG16Net(img_rows,img_cols,img_channels,nb_classes, 1e-6)
VGG16_model.summary()
VGG16_model.compile(SGD(lr=0.01, momentum=0.9, decay=0.00001, nesterov=False),
        loss = 'categorical_crossentropy',
        metrics=['accuracy'])
#clc = CyclicLR(base_lr= 0.01, max_lr=0.1, step_size = 2*X_train.shape[0]//n_batch, mode = 'triangular')


if not data_aug:
    print('Train without data augmentation!')
    VGG16_model.fit(
        X_train,
        Y_train,
        batch_size=n_batch,
        epochs=n_epochs,
        validation_data=(X_test, Y_test),
        shuffle=True,
        verbose=1,
        callbacks=[lrf])
else:
    print('Train with data augmentation!')
    img_gen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=True,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        # randomly rotate images in the range (degrees, 0 to 180)
        rotation_range=90,
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True) # randomly flip images
    img_gen.fit(X_train)

    VGG16_model.fit_generator(img_gen.flow(X_train, Y_train, batch_size = n_batch, shuffle = True),
        steps_per_epoch = X_train.shape[0] // n_batch,
        validation_data = (X_test, Y_test),
        epochs = n_epochs,
        verbose = 1,
        callbacks=[lrf])
lrf.plot_schedule(clip_beginning=10, clip_endding=5)

scores = VGG16_model.evaluate(X_test, Y_test, batch_size=n_batch)
for score, metric_name in zip(scores, VGG16_model.metrics_names):
    print("%s : %0.4f" % (metric_name, score))

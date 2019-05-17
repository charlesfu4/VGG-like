from __future__ import print_function
import numpy as np
import keras
from keras import backend as K 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from VGG16 import VGG16Net
from clr_callback import CyclicLR
from clr import OneCycleLR
#Load oxflower17 dataset
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

x, y = oxflower17.load_data(one_hot=True)

#Split train and test data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2,shuffle = True)

#Call VGG16Net model
# input image dimensions
data_aug = True
img_rows, img_cols = 224,224
# The CIFAR10 images are RGB.
img_channels = 3
nb_classes = 17
n_epochs = 10
n_batch = 8
# The data, shuffled and split between train and test sets:

# Convert class vectors to binary class matrices.


# preprocess input
# normalization
VGG16_model = VGG16Net(img_rows,img_cols,img_channels,17, 1e-5)
VGG16_model.summary()
VGG16_model.compile(SGD(lr=0.00407, momentum=0.9, decay=0.00001, nesterov=False),
        loss = 'categorical_crossentropy',
        metrics=['accuracy'])
#clc = CyclicLR(base_lr= 0.00407, max_lr=0.01023, step_size = 4*X_train.shape[0]//n_batch, mode = 'triangular')
clc = OneCycleLR(max_lr=0.1023,
        maximum_momentum = 0.9,
        end_percentage=0.1,
        verbose=True)


if ~data_aug:
    History = VGG16_model.fit(
            X_train,
            Y_train,
            batch_size=n_batch,
            epochs=n_epochs,
            validation_data=(X_test, Y_test),
            shuffle=True,
            verbose=1,
            callbacks=[clc,LRTensorBoard(log_dir='./Graph/6')])

#Start training using dataaugumentation generator
else:
    img_gen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=True,  # divide each input by its std
        zca_whitening=10,  # apply ZCA whitening
        # randomly rotate images in the range (degrees, 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False) # randomly flip images
    img_gen.fit(X_train)

    History = VGG16_model.fit_generator(img_gen.flow(X_train, Y_train, batch_size = n_batch, shuffle = True),
        steps_per_epoch = X_train.shape[0] // n_batch, 
        validation_data = (X_test, Y_test), 
        epochs = n_epoch,
        verbose = 1, 
        callbacks=[clc,LRTensorBoard(log_dir='./Graph/5')])

#from __future__ import print_function
import numpy as np
import keras
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from VGG16 import VGG16Net
from clr_callback import CyclicLR
from keras.callbacks import TensorBoard
from clr import OneCycleLR
#Tensorboard callback for cyclical learning rate
class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


# input image dimensions
data_aug = True
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3
nb_classes = 10
n_epochs = 10
n_batch = 128
# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# preprocess input
# normalization
mean = np.mean(X_train, axis=(0, 1, 2), keepdims=True).astype('float32')
std = np.mean(X_train, axis=(0, 1, 2), keepdims=True).astype('float32')

print("Channel Mean : ", mean)
print("Channel Std : ", std)

X_train = (X_train - mean) / (std)
X_test = (X_test - mean) / (std)    
VGG16_model = VGG16Net(img_rows,img_cols,img_channels,10, 1e-5)
VGG16_model.summary()
VGG16_model.compile(SGD(lr=0.0173, momentum=0.9, decay=0, nesterov=True),
        loss = 'categorical_crossentropy',
        metrics=['accuracy'])
#clc = CyclicLR(base_lr= 0.003, max_lr=0.1, step_size = 5*X_train.shape[0]//n_batch, mode = 'triangular')
clc = OneCycleLR(max_lr = 0.15,
        end_percentage = 0.1,
        verbose = True)

tb = TensorBoard(log_dir='./Graph/6',
        histogram_freq=0,
        write_graph=True, 
        write_images=True)

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
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        # randomly rotate images in the range (degrees, 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False) # randomly flip images
    img_gen.fit(X_train)

    VGG16_model.fit_generator(img_gen.flow(X_train, Y_train, batch_size = n_batch, shuffle = True),
        steps_per_epoch = X_train.shape[0] // n_batch, 
        validation_data = (X_test, Y_test), 
        epochs = n_epoch,
        verbose = 1, 
        callbacks=[clc,LRTensorBoard(log_dir='./Graph6')])
#Plot Loss and Accuracy
#while (True):
#    plt.figure(figsize = (15,5))
#    plt.subplot(1,2,1)
#
#    plt.plot(History.history['acc'])
#    plt.plot(History.history['val_acc'])
#    plt.title('model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
#    plt.show()
#    
#    plt.subplot(1,2,2)
#    plt.plot(History.history['loss'])
#    plt.plot(History.history['val_loss'])
#    plt.title('model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')

    #plt.subplot(1,3,3)
    #plt.plot(clc.history['lr'])
    #plt.title('model learning rate')
    #plt.ylabel('learning')
    #plt.xlabel('iteration')
    #plt.legend(['learning rate'], loc='upper left')
    #plt.show()

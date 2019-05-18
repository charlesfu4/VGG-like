import numpy as np
import keras
from keras import regularizers
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD



def VGG16Net(width,
        height,
        depth,
        classes,
        weight_decay
        ):
    
    model = Sequential()
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(width,height,depth),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
   
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    
    
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))   
    model.add(BatchNormalization())
    
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(4096,activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))

    model.add(Dense(4096,activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))

    model.add(Dense(1000,activation='relu',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes,activation='softmax'))
    
    return model

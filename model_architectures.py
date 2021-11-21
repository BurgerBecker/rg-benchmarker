from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import MinMaxScaler
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras
from tensorflow.compat.v1.keras.models import Sequential, Model
from tensorflow.compat.v1.keras.layers import Activation, Add, Conv2D, Dense, Dropout, Flatten, Input, LeakyReLU, MaxPooling2D, BatchNormalization
# from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.compat.v1.keras import backend as K
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.compat.v1.keras.optimizers import Adadelta as adadelta
from tensorflow.compat.v1.keras.losses import categorical_crossentropy as cce
# import tensorflow as tf
from sklearn import metrics
from tensorflow.compat.v1.keras.optimizers import RMSprop
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage.transform import rotate
from tensorflow.compat.v1.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.compat.v1.keras.models import load_model
from tensorflow.compat.v1.keras import regularizers
import random
import os
from tensorflow.compat.v1.keras import initializers

def alexnet(random_state,input_shape,num_classes):
    # Keras implementation from:
    # https://www.mydatahack.com/building-alexnet-with-keras/
    
    # (3) Create a sequential model
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Dense Layer
    model.add(Dense(1000, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    # Output Layer
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def alexnet_erf(random_state,input_shape,num_classes):
    # Keras implementation from:
    # https://www.mydatahack.com/building-alexnet-with-keras/
    
    # (3) Create a sequential model
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(name="conv2d_input",activation='relu',filters=96, kernel_size=(11,11), strides=(4,4), padding='valid',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    # model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256,activation='relu', kernel_size=(5,5), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    # Batch Normalisation
    # model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Batch Normalisation
    # model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Batch Normalisation
    # model.add(BatchNormalization())
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid',name="final_output"))
    # Batch Normalisation
    # model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    # model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    # model.add(BatchNormalization())

    # 3rd Dense Layer
    # model.add(Dense(1000, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # # Add Dropout
    # model.add(Dropout(0.4))
    # Batch Normalisation
    # model.add(BatchNormalization())
    # Output Layer
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def alexnet_real(random_state,input_shape,num_classes):
    # Keras implementation from:
    # https://www.mydatahack.com/building-alexnet-with-keras/
    
    # (3) Create a sequential model
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(name="conv2d_input",activation='relu',filters=96, kernel_size=(11,11), strides=(4,4), padding='valid',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    # model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256,activation='relu', kernel_size=(5,5), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    # Batch Normalisation
    # model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Batch Normalisation
    # model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Batch Normalisation
    # model.add(BatchNormalization())
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid',name="final_output"))
    # Batch Normalisation
    # model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Dense Layer
    # model.add(Dense(1000, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # # Add Dropout
    # model.add(Dropout(0.4))
    # Batch Normalisation
    # model.add(BatchNormalization())
    # Output Layer
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def alexnet_real_regularized(random_state,input_shape,num_classes):
    # Keras implementation from:
    # https://www.mydatahack.com/building-alexnet-with-keras/
    
    # (3) Create a sequential model
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(name="conv2d_input",activation='relu',filters=96, kernel_size=(11,11), strides=(4,4), padding='valid',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    # model.add(BatchNormalization())
    model.add(Dropout(0.25,seed=random_state))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256,activation='relu', kernel_size=(5,5), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    # Batch Normalisation
    # model.add(BatchNormalization())
    model.add(Dropout(0.25,seed=random_state))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Batch Normalisation
    # model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Batch Normalisation
    # model.add(BatchNormalization())
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid',name="final_output"))
    # Batch Normalisation
    # model.add(BatchNormalization())
    model.add(Dropout(0.25,seed=random_state))

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4,seed=random_state))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4,seed=random_state))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Dense Layer
    # model.add(Dense(1000, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # # Add Dropout
    # model.add(Dropout(0.4))
    # Batch Normalisation
    # model.add(BatchNormalization())
    # Output Layer
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model


def alexnet_real_do_seed(random_state,input_shape,num_classes):
    # Keras implementation from:
    # https://www.mydatahack.com/building-alexnet-with-keras/
    
    # (3) Create a sequential model
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(name="conv2d_input",activation='relu',filters=96, kernel_size=(11,11), strides=(4,4), padding='valid',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    # model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256,activation='relu', kernel_size=(5,5), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    # Batch Normalisation
    # model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Batch Normalisation
    # model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Batch Normalisation
    # model.add(BatchNormalization())
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid',name="final_output"))
    # Batch Normalisation
    # model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4,seed=random_state))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4,seed=random_state))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Dense Layer
    # model.add(Dense(1000, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # # Add Dropout
    # model.add(Dropout(0.4))
    # Batch Normalisation
    # model.add(BatchNormalization())
    # Output Layer
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def alexnet_adapted(random_state,input_shape,num_classes):
    # Keras implementation from:
    # https://www.mydatahack.com/building-alexnet-with-keras/
    
    # (3) Create a sequential model
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(name="conv2d_input",activation='relu',filters=96, kernel_size=(11,11), strides=(4,4), padding='valid',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256,activation='relu', kernel_size=(5,5), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid',name="final_output"))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4,seed=random_state))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4,seed=random_state))
    # Batch Normalisation
    model.add(BatchNormalization())
    # Output Layer
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def alexnet_real_do_seed_same(random_state,input_shape,num_classes):
    # Keras implementation from:
    # https://www.mydatahack.com/building-alexnet-with-keras/
    
    # (3) Create a sequential model
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(name="conv2d_input",activation='relu',filters=96, kernel_size=(11,11), strides=(4,4), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))
    # Batch Normalisation before passing it to the next layer
    # model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256,activation='relu', kernel_size=(5,5), strides=(1,1), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))
    # Batch Normalisation
    # model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384,activation='relu', kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Batch Normalisation
    # model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384,activation='relu', kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Batch Normalisation
    # model.add(BatchNormalization())
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256,activation='relu', kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same',name="final_output"))
    # Batch Normalisation
    # model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4,seed=random_state))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4,seed=random_state))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Dense Layer
    # model.add(Dense(1000, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # # Add Dropout
    # model.add(Dropout(0.4))
    # Batch Normalisation
    # model.add(BatchNormalization())
    # Output Layer
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def alexnet_real_bn(random_state,input_shape,num_classes):
    
    # (3) Create a sequential model
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(name="conv2d_input",activation='relu',filters=96, kernel_size=(11,11), strides=(4,4), padding='valid',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling 
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256,activation='relu', kernel_size=(5,5), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256,activation='relu', kernel_size=(3,3), strides=(1,1), padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid',name="final_output"))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    # Output Layer
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def mcrgnet(random_state,input_shape,num_classes):
    # Model name: MCRGNet (Morphological Classification of Radio Galaxy Network) 
    # Title: A Machine Learning Based Morphological Classification of 14,245 Radio AGN's Selected from the Best-Heckman Sample
    # Links to Article:
    # https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2
    # https://arxiv.org/pdf/1812.07190.pdf
    # Publication date: 5 Feb 2019
    # Primary Author: Zhixian Ma

    # Architecture description: (Page 5, figure 2 b) https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2/pdf#page=5

    # Source code link:
    # https://github.com/myinxd/MCRGNet/blob/master/mcrgnet/ConvAE.py

    # Notes: Although the original article used an autoencoder for pretraining the CNN, we only train the CNN architecture described.
    # Our reason for this is computational constraints on retraining 10 times on different data splits in the original fashion. 
    # The original training strategy used is quite interesting and we recommend that the reader investigate the article.

    # The architecture defined here is adapted from the article description and not the Github repo
    # for the purpose of clarity.
    
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3,3), strides = 2, activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(8, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.25, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def mcrgnet_no_do(random_state,input_shape,num_classes):
    # Model name: MCRGNet (Morphological Classification of Radio Galaxy Network) 
    # Title: A Machine Learning Based Morphological Classification of 14,245 Radio AGN's Selected from the Best-Heckman Sample
    # Links to Article:
    # https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2
    # https://arxiv.org/pdf/1812.07190.pdf
    # Publication date: 5 Feb 2019
    # Primary Author: Zhixian Ma

    # Architecture description: (Page 5, figure 2 b) https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2/pdf#page=5

    # Source code link:
    # https://github.com/myinxd/MCRGNet/blob/master/mcrgnet/ConvAE.py

    # Notes: Although the original article used an autoencoder for pretraining the CNN, we only train the CNN architecture described.
    # Our reason for this is computational constraints on retraining 10 times on different data splits in the original fashion. 
    # The original training strategy used is quite interesting and we recommend that the reader investigate the article.

    # The architecture defined here is adapted from the article description and not the Github repo
    # for the purpose of clarity.
    
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3,3), strides = 2, activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(8, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')) 
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def mcrgnet_no_conv_do(random_state,input_shape,num_classes):
    # Model name: MCRGNet (Morphological Classification of Radio Galaxy Network) 
    # Title: A Machine Learning Based Morphological Classification of 14,245 Radio AGN's Selected from the Best-Heckman Sample
    # Links to Article:
    # https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2
    # https://arxiv.org/pdf/1812.07190.pdf
    # Publication date: 5 Feb 2019
    # Primary Author: Zhixian Ma

    # Architecture description: (Page 5, figure 2 b) https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2/pdf#page=5

    # Source code link:
    # https://github.com/myinxd/MCRGNet/blob/master/mcrgnet/ConvAE.py

    # Notes: Although the original article used an autoencoder for pretraining the CNN, we only train the CNN architecture described.
    # Our reason for this is computational constraints on retraining 10 times on different data splits in the original fashion. 
    # The original training strategy used is quite interesting and we recommend that the reader investigate the article.

    # The architecture defined here is adapted from the article description and not the Github repo
    # for the purpose of clarity.
    
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3,3), strides = 2, activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(8, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.25, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def mcrgnet_linear_no_do(random_state,input_shape,num_classes):
    # Model name: MCRGNet (Morphological Classification of Radio Galaxy Network) 
    # Title: A Machine Learning Based Morphological Classification of 14,245 Radio AGN's Selected from the Best-Heckman Sample
    # Links to Article:
    # https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2
    # https://arxiv.org/pdf/1812.07190.pdf
    # Publication date: 5 Feb 2019
    # Primary Author: Zhixian Ma

    # Architecture description: (Page 5, figure 2 b) https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2/pdf#page=5

    # Source code link:
    # https://github.com/myinxd/MCRGNet/blob/master/mcrgnet/ConvAE.py

    # Notes: Although the original article used an autoencoder for pretraining the CNN, we only train the CNN architecture described.
    # Our reason for this is computational constraints on retraining 10 times on different data splits in the original fashion. 
    # The original training strategy used is quite interesting and we recommend that the reader investigate the article.

    # The architecture defined here is adapted from the article description and not the Github repo
    # for the purpose of clarity.
    
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3,3), strides = 2, activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(8, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='linear', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.25, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def mcrgnet_l2_no_do(random_state,input_shape,num_classes):
    # Model name: MCRGNet (Morphological Classification of Radio Galaxy Network) 
    # Title: A Machine Learning Based Morphological Classification of 14,245 Radio AGN's Selected from the Best-Heckman Sample
    # Links to Article:
    # https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2
    # https://arxiv.org/pdf/1812.07190.pdf
    # Publication date: 5 Feb 2019
    # Primary Author: Zhixian Ma

    # Architecture description: (Page 5, figure 2 b) https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2/pdf#page=5

    # Source code link:
    # https://github.com/myinxd/MCRGNet/blob/master/mcrgnet/ConvAE.py

    # Notes: Although the original article used an autoencoder for pretraining the CNN, we only train the CNN architecture described.
    # Our reason for this is computational constraints on retraining 10 times on different data splits in the original fashion. 
    # The original training strategy used is quite interesting and we recommend that the reader investigate the article.

    # The architecture defined here is adapted from the article description and not the Github repo
    # for the purpose of clarity.
    
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3,3), strides = 2, activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(8, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.25, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def mcrgnet_l2_do(random_state,input_shape,num_classes):
    # Model name: MCRGNet (Morphological Classification of Radio Galaxy Network) 
    # Title: A Machine Learning Based Morphological Classification of 14,245 Radio AGN's Selected from the Best-Heckman Sample
    # Links to Article:
    # https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2
    # https://arxiv.org/pdf/1812.07190.pdf
    # Publication date: 5 Feb 2019
    # Primary Author: Zhixian Ma

    # Architecture description: (Page 5, figure 2 b) https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2/pdf#page=5

    # Source code link:
    # https://github.com/myinxd/MCRGNet/blob/master/mcrgnet/ConvAE.py

    # Notes: Although the original article used an autoencoder for pretraining the CNN, we only train the CNN architecture described.
    # Our reason for this is computational constraints on retraining 10 times on different data splits in the original fashion. 
    # The original training strategy used is quite interesting and we recommend that the reader investigate the article.

    # The architecture defined here is adapted from the article description and not the Github repo
    # for the purpose of clarity.
    
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3,3), strides = 2, activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(8, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.25, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def mcrgnet_regularized(random_state,input_shape,num_classes):
    # Model name: MCRGNet (Morphological Classification of Radio Galaxy Network) 
    # Title: A Machine Learning Based Morphological Classification of 14,245 Radio AGN's Selected from the Best-Heckman Sample
    # Links to Article:
    # https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2
    # https://arxiv.org/pdf/1812.07190.pdf
    # Publication date: 5 Feb 2019
    # Primary Author: Zhixian Ma

    # Architecture description: (Page 5, figure 2 b) https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2/pdf#page=5

    # Source code link:
    # https://github.com/myinxd/MCRGNet/blob/master/mcrgnet/ConvAE.py

    # Notes: Although the original article used an autoencoder for pretraining the CNN, we only train the CNN architecture described.
    # Our reason for this is computational constraints on retraining 10 times on different data splits in the original fashion. 
    # The original training strategy used is quite interesting and we recommend that the reader investigate the article.

    # The architecture defined here is adapted from the article description and not the Github repo
    # for the purpose of clarity.
    
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3,3), strides = 2, activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(8, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def mcrgnet_linear(random_state,input_shape,num_classes):
    # Model name: MCRGNet (Morphological Classification of Radio Galaxy Network) 
    # Title: A Machine Learning Based Morphological Classification of 14,245 Radio AGN's Selected from the Best-Heckman Sample
    # Links to Article:
    # https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2
    # https://arxiv.org/pdf/1812.07190.pdf
    # Publication date: 5 Feb 2019
    # Primary Author: Zhixian Ma

    # Architecture description: (Page 5, figure 2 b) https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2/pdf#page=5

    # Source code link:
    # https://github.com/myinxd/MCRGNet/blob/master/mcrgnet/ConvAE.py

    # Notes: Although the original article used an autoencoder for pretraining the CNN, we only train the CNN architecture described.
    # Our reason for this is computational constraints on retraining 10 times on different data splits in the original fashion. 
    # The original training strategy used is quite interesting and we recommend that the reader investigate the article.

    # The architecture defined here is adapted from the article description and not the Github repo
    # for the purpose of clarity.
    
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3,3), strides = 2, activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(8, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def mcrgnet_relu(random_state,input_shape,num_classes):
    # Model name: MCRGNet (Morphological Classification of Radio Galaxy Network) 
    # Title: A Machine Learning Based Morphological Classification of 14,245 Radio AGN's Selected from the Best-Heckman Sample
    # Links to Article:
    # https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2
    # https://arxiv.org/pdf/1812.07190.pdf
    # Publication date: 5 Feb 2019
    # Primary Author: Zhixian Ma

    # Architecture description: (Page 5, figure 2 b) https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2/pdf#page=5

    # Source code link:
    # https://github.com/myinxd/MCRGNet/blob/master/mcrgnet/ConvAE.py

    # Notes: Although the original article used an autoencoder for pretraining the CNN, we only train the CNN architecture described.
    # Our reason for this is computational constraints on retraining 10 times on different data splits in the original fashion. 
    # The original training strategy used is quite interesting and we recommend that the reader investigate the article.

    # The architecture defined here is adapted from the article description and not the Github repo
    # for the purpose of clarity.
    
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3,3), strides = 2, activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(8, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def mcrgnet_linear_less_do(random_state,input_shape,num_classes):
    # Model name: MCRGNet (Morphological Classification of Radio Galaxy Network) 
    # Title: A Machine Learning Based Morphological Classification of 14,245 Radio AGN's Selected from the Best-Heckman Sample
    # Links to Article:
    # https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2
    # https://arxiv.org/pdf/1812.07190.pdf
    # Publication date: 5 Feb 2019
    # Primary Author: Zhixian Ma

    # Architecture description: (Page 5, figure 2 b) https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2/pdf#page=5

    # Source code link:
    # https://github.com/myinxd/MCRGNet/blob/master/mcrgnet/ConvAE.py

    # Notes: Although the original article used an autoencoder for pretraining the CNN, we only train the CNN architecture described.
    # Our reason for this is computational constraints on retraining 10 times on different data splits in the original fashion. 
    # The original training strategy used is quite interesting and we recommend that the reader investigate the article.

    # The architecture defined here is adapted from the article description and not the Github repo
    # for the purpose of clarity.
    
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3,3), strides = 2, activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(8, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def mcrgnet_erf(random_state,input_shape,num_classes):
    # Model name: MCRGNet (Morphological Classification of Radio Galaxy Network) 
    # Title: A Machine Learning Based Morphological Classification of 14,245 Radio AGN's Selected from the Best-Heckman Sample
    # Links to Article:
    # https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2
    # https://arxiv.org/pdf/1812.07190.pdf
    # Publication date: 5 Feb 2019
    # Primary Author: Zhixian Ma

    # Architecture description: (Page 5, figure 2 b) https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2/pdf#page=5

    # Source code link:
    # https://github.com/myinxd/MCRGNet/blob/master/mcrgnet/ConvAE.py

    # Notes: Although the original article used an autoencoder for pretraining the CNN, we only train the CNN architecture described.
    # Our reason for this is computational constraints on retraining 10 times on different data splits in the original fashion. 
    # The original training strategy used is quite interesting and we recommend that the reader investigate the article.

    # The architecture defined here is adapted from the article description and not the Github repo
    # for the purpose of clarity.
    
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3,3), strides = 2, activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(8, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.25, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def mcrgnet_larger(random_state,input_shape,num_classes):
    # Model name: MCRGNet (Morphological Classification of Radio Galaxy Network) 
    # Title: A Machine Learning Based Morphological Classification of 14,245 Radio AGN's Selected from the Best-Heckman Sample
    # Links to Article:
    # https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2
    # https://arxiv.org/pdf/1812.07190.pdf
    # Publication date: 5 Feb 2019
    # Primary Author: Zhixian Ma

    # Architecture description: (Page 5, figure 2 b) https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2/pdf#page=5

    # Source code link:
    # https://github.com/myinxd/MCRGNet/blob/master/mcrgnet/ConvAE.py

    # Notes: Although the original article used an autoencoder for pretraining the CNN, we only train the CNN architecture described.
    # Our reason for this is computational constraints on retraining 10 times on different data splits in the original fashion. 
    # The original training strategy used is quite interesting and we recommend that the reader investigate the article.

    # The architecture defined here is adapted from the article description and not the Github repo
    # for the purpose of clarity.
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3,3), strides = 2, activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), strides = 2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.25, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def simple_no_do(random_state,input_shape,num_classes):
    # Model name: SimpleNet
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: V. Lukic

    # Architecture description: Not in the paper, only in the repo

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    # Note: This model is not mentioned in the paper but is available in the Github repo.
    # Since it contained non-standard 4 by 4 kernel sizes, this warrented inclusion to the study.
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(16, (4, 4), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(16, (4, 4), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((4, 4),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (4, 4), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(16, (4, 4), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same', name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def simple(random_state,input_shape,num_classes):
    # Model name: SimpleNet
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: V. Lukic

    # Architecture description: Not in the paper, only in the repo

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    # Note: This model is not mentioned in the paper but is available in the Github repo.
    # Since it contained non-standard 4 by 4 kernel sizes, this warrented inclusion to the study.
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(16, (4, 4), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(16, (4, 4), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((4, 4),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (4, 4), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(16, (4, 4), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same', name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def simple_erf(random_state,input_shape,num_classes):
    # Model name: SimpleNet
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: V. Lukic

    # Architecture description: Not in the paper, only in the repo

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    # Note: This model is not mentioned in the paper but is available in the Github repo.
    # Since it contained non-standard 4 by 4 kernel sizes, this warrented inclusion to the study.
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(16, (4, 4), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(16, (4, 4), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((4, 4),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(16, (4, 4), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(16, (4, 4), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same', name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def conv4_no_reg(random_state,input_shape,num_classes):
    # Model name: ConvNet 4
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: V. Lukic

    # Architecture description: (Page 8, table 3) https://arxiv.org/pdf/1905.03274.pdf#page=8

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(16, (5, 5),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(16, (5, 5), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(16, (5, 5), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Flatten())
    model.add(Dense(500, activation='linear', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
    # model.add(Dropout(0.5, seed=random_state))   
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def conv4(random_state,input_shape,num_classes):
    # Model name: ConvNet 4
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: V. Lukic

    # Architecture description: (Page 8, table 3) https://arxiv.org/pdf/1905.03274.pdf#page=8

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(16, (5, 5),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(16, (5, 5), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(16, (5, 5), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_state))   
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def conv4_no_do(random_state,input_shape,num_classes):
    # Model name: ConvNet 4
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: V. Lukic

    # Architecture description: (Page 8, table 3) https://arxiv.org/pdf/1905.03274.pdf#page=8

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(16, (5, 5),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(16, (5, 5), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(16, (5, 5), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_state))   
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def conv4_l2_no_do(random_state,input_shape,num_classes):
    # Model name: ConvNet 4
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: V. Lukic

    # Architecture description: (Page 8, table 3) https://arxiv.org/pdf/1905.03274.pdf#page=8

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(16, (5, 5),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(16, (5, 5), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(16, (5, 5), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Flatten())
    model.add(Dense(500, activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_state))   
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def conv4_l2_do(random_state,input_shape,num_classes):
    # Model name: ConvNet 4
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: V. Lukic

    # Architecture description: (Page 8, table 3) https://arxiv.org/pdf/1905.03274.pdf#page=8

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(16, (5, 5),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(16, (5, 5), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(16, (5, 5), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Flatten())
    model.add(Dense(500, activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_state))   
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def conv4_erf(random_state,input_shape,num_classes):
    # Model name: ConvNet 4
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: V. Lukic

    # Architecture description: (Page 8, table 3) https://arxiv.org/pdf/1905.03274.pdf#page=8

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(16, (5, 5),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(16, (5, 5), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(16, (5, 5), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
    # model.add(Dropout(0.5, seed=random_state))   
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def conv8(random_state,input_shape,num_classes):
    # Model name: ConvNet 8
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: Vesna Lukic

    # Architecture description: (Page 8, table 3) https://arxiv.org/pdf/1905.03274.pdf#page=9

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros',name="output"))
    return model

def conv8_relu(random_state,input_shape,num_classes):
    # Model name: ConvNet 8
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: Vesna Lukic

    # Architecture description: (Page 8, table 3) https://arxiv.org/pdf/1905.03274.pdf#page=9

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='relu', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros',name="output"))
    return model

def conv8_full_do(random_state,input_shape,num_classes):
    # Model name: ConvNet 8
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: Vesna Lukic

    # Architecture description: (Page 8, table 3) https://arxiv.org/pdf/1905.03274.pdf#page=9

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros',name="output"))
    return model

def conv8_half_do(random_state,input_shape,num_classes):
    # Model name: ConvNet 8
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: Vesna Lukic

    # Architecture description: (Page 8, table 3) https://arxiv.org/pdf/1905.03274.pdf#page=9

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros',name="output"))
    return model

def conv8_no_reg(random_state,input_shape,num_classes):
    # Model name: ConvNet 8
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: Vesna Lukic

    # Architecture description: (Page 8, table 3) https://arxiv.org/pdf/1905.03274.pdf#page=9

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros',name="output"))
    return model

def conv8_no_seed(random_state,input_shape,num_classes):
    # Model name: ConvNet 8
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: Vesna Lukic

    # Architecture description: (Page 8, table 3) https://arxiv.org/pdf/1905.03274.pdf#page=9

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros',name="output"))
    return model

def conv8_xpress(random_state,input_shape,num_classes):
    # Model name: ConvNet 8
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: Vesna Lukic

    # Architecture description: (Page 8, table 3) https://arxiv.org/pdf/1905.03274.pdf#page=9

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=2,activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros',name="output"))
    return model

def conv8_erf(random_state,input_shape,num_classes):
    # Model name: ConvNet 8
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: Vesna Lukic

    # Architecture description: (Page 8, table 3) https://arxiv.org/pdf/1905.03274.pdf#page=9

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    # model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros',name="output"))
    return model

def hosenie_big(random_state,input_shape,num_classes):
    # Source classification in deep radio surveys using machine learning techniques
    # Primary Author: Zafiirah Banon Hosenie
    # http://hdl.handle.net/10394/31250
    # Architecture description: Page 97
    # Date of thesis hand-in: 2017-11-20
    
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(11,11), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(19, kernel_size=(5,5), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(128, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(40, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dense(40, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Activation('softmax'))
    return model

def conv8_final(random_state,input_shape,num_classes):
    # Model name: ConvNet 8
    # Title: Morphological classification of radio galaxies: capsule networks versus convolutional neural networks
    # Links to article:
    # https://academic.oup.com/mnras/article-abstract/487/2/1729/5492264?redirectedFrom=fulltext
    # https://arxiv.org/pdf/1905.03274.pdf
    # Publication date: August 2019
    # Primary Author: Vesna Lukic

    # Architecture description: (Page 8, table 3) https://arxiv.org/pdf/1905.03274.pdf#page=9

    # Source code link:
    # https://github.com/vlukic973/RadioGalaxy_Conv_Caps/blob/master/convnet_LOFAR_radio_galaxy.py
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def toothless_real(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11,11), padding='valid', strides=4,input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=2,padding='valid'))
    model.add(Conv2D(256, (5, 5), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    model.add(Flatten())
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))  
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def toothless_no_do_no_bn(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11,11), padding='valid', strides=4,input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=2,padding='valid'))
    model.add(Conv2D(256, (5, 5), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    model.add(Flatten())
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))  
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def toothless_linear_no_do(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11,11), padding='valid', strides=4,input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=2,padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (5, 5), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(4096, kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('linear'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))  
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def toothless_l2_no_do(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11,11), padding='valid', strides=4,input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=2,padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (5, 5), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(4096, kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))  
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model


def toothless_linear(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11,11), padding='valid', strides=4,input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=2,padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (5, 5), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(4096, kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('linear'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))  
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def toothless_l2_do(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11,11), padding='valid', strides=4,input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=2,padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (5, 5), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(4096, kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))  
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def toothless_reg_no_bn(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11,11), padding='valid', strides=4,input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=2,padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (5, 5), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))  
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def toothless_erf(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11,11), padding='valid', strides=4,input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=2,padding='valid'))
    model.add(Conv2D(256, (5, 5), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid'))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='valid',name='final_output'))
    model.add(Flatten())
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5, seed=random_state))  
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def cosmodeep(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))    
    model.add(MaxPooling2D((2, 2),padding='same'))    
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def radio_galaxy_zoo(random_state,input_shape,num_classes):
    # Title: Radio Galaxy Zoo: compact and extended radio source classification with deep learning
    # https://academic.oup.com/mnras/article/476/1/246/4826039
    # Primary Author: Vesna Lukic
    # Publication date: 26 Jan 2018

    # Notes: We assume that the padding method used is same padding
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(8, 8), strides=3, padding='same', input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (7, 7), strides=2, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, (2, 2),strides=1, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2),name='final_output'))
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(1024, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))  
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def radio_galaxy_zoo_regularized(random_state,input_shape,num_classes):
    # Title: Radio Galaxy Zoo: compact and extended radio source classification with deep learning
    # https://academic.oup.com/mnras/article/476/1/246/4826039
    # Primary Author: Vesna Lukic
    # Publication date: 26 Jan 2018

    # Notes: We assume that the padding method used is same padding
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(8, 8), strides=3, padding='same', input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (7, 7), strides=2, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (2, 2),strides=1, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2),name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(1024, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))  
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def radio_galaxy_zoo_same(random_state,input_shape,num_classes):
    # Title: Radio Galaxy Zoo: compact and extended radio source classification with deep learning
    # https://academic.oup.com/mnras/article/476/1/246/4826039
    # Primary Author: Vesna Lukic
    # Publication date: 26 Jan 2018

    # Notes: We assume that the padding method used is same padding
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(8, 8), strides=3, padding='same', input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (7, 7), strides=2, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))
    model.add(Conv2D(64, (2, 2),strides=1, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same',name='final_output'))
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(1024, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))  
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def radio_galaxy_zoo_conv_blocks(random_state,input_shape,num_classes):
	#Modified version in which larger layers are split into smaller ones

    # Title: Radio Galaxy Zoo: compact and extended radio source classification with deep learning
    # https://academic.oup.com/mnras/article/476/1/246/4826039
    # Primary Author: Vesna Lukic
    # Publication date: 26 Jan 2018

    # Notes: We assume that the padding method used is same padding
    
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3), strides=3, padding='same', input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), strides=1, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), strides=2, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), strides=1, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, (2, 2),strides=1, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2),name='final_output'))
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(1024, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))  
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def radio_galaxy_zoo_mod(random_state,input_shape,num_classes):
    # Title: Radio Galaxy Zoo: compact and extended radio source classification with deep learning
    # https://academic.oup.com/mnras/article/476/1/246/4826039
    # Primary Author: Vesna Lukic
    # Publication date: 26 Jan 2018

    # Notes: We assume that the padding method used is same padding 
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(8, 8), strides=2, padding='same', input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (7, 7), strides=1, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, (2, 2),strides=1, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2),name='final_output'))
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(1024, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5, seed=random_state))  
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def radio_galaxy_zoo_linear(random_state,input_shape,num_classes):
    # Title: Radio Galaxy Zoo: compact and extended radio source classification with deep learning
    # https://academic.oup.com/mnras/article/476/1/246/4826039
    # Primary Author: Vesna Lukic
    # Publication date: 26 Jan 2018

    # Notes: We assume that the padding method used is same padding 
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(8, 8), strides=3, padding='same', input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (7, 7), strides=2, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, (2, 2),strides=1, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def conv8_relu(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def fr_deep(random_state,input_shape,num_classes):
    # Transfer learning for radio galaxy classification
    # Primary Author: Hongming Tang
    # https://github.com/HongmingTang060313/FR-DEEP
    # https://academic.oup.com/mnras/article/488/3/3358/5538844
    # https://arxiv.org/pdf/1903.11921.pdf
    # Architecture description: Page 7
    # 25 July 2019
    # Notes: This is the original architecture
    
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(11,11),strides=1,padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    model.add(Conv2D(16, (5, 5),strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),strides=3))
    model.add(Conv2D(24, (3, 3),strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(24, (3, 3),strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3),strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(5,5),strides=5))
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def fr_deep_regularized(random_state,input_shape,num_classes):
    # Transfer learning for radio galaxy classification
    # Primary Author: Hongming Tang
    # https://github.com/HongmingTang060313/FR-DEEP
    # https://academic.oup.com/mnras/article/488/3/3358/5538844
    # https://arxiv.org/pdf/1903.11921.pdf
    # Architecture description: Page 7
    # 25 July 2019
    # Notes: This is the original architecture
    
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(11,11),strides=1, padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='valid'))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(16, (5, 5),strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=3, padding='valid'))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(24, (3, 3), strides=1, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(24, (3, 3), strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(16, (3, 3), strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(5,5), strides=5, padding='valid'))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def fr_deep_dropout(random_state,input_shape,num_classes):
    # Transfer learning for radio galaxy classification
    # Primary Author: Hongming Tang
    # https://github.com/HongmingTang060313/FR-DEEP
    # https://academic.oup.com/mnras/article/488/3/3358/5538844
    # https://arxiv.org/pdf/1903.11921.pdf
    # Architecture description: Page 7
    # 25 July 2019
    # Notes: This is the original architecture
    
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(11,11),strides=1, padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25,seed=random_state))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='valid'))
    model.add(Conv2D(16, (5, 5),strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25,seed=random_state))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=3, padding='valid'))
    model.add(Conv2D(24, (3, 3), strides=1, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(24, (3, 3), strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(16, (3, 3), strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25,seed=random_state))
    model.add(MaxPooling2D(pool_size=(5,5), strides=5, padding='valid'))
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def fr_deep_no_bn(random_state,input_shape,num_classes):
    # Transfer learning for radio galaxy classification
    # Primary Author: Hongming Tang
    # https://github.com/HongmingTang060313/FR-DEEP
    # https://academic.oup.com/mnras/article/488/3/3358/5538844
    # https://arxiv.org/pdf/1903.11921.pdf
    # Architecture description: Page 7
    # 25 July 2019
    # Notes: This is the original architecture
    
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(11,11),strides=1, padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='valid'))
    model.add(Conv2D(16, (5, 5),strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=3, padding='valid'))
    model.add(Conv2D(24, (3, 3), strides=1, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(24, (3, 3), strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(5,5), strides=5, padding='valid'))
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def fr_deep_real(random_state,input_shape,num_classes):
    # Transfer learning for radio galaxy classification
    # Primary Author: Hongming Tang
    # https://github.com/HongmingTang060313/FR-DEEP
    # https://academic.oup.com/mnras/article/488/3/3358/5538844
    # https://arxiv.org/pdf/1903.11921.pdf
    # Architecture description: Page 7
    # 25 July 2019
    # Notes: This is the original architecture
    
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(11,11),strides=1, padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='valid'))
    model.add(Conv2D(16, (5, 5),strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=3, padding='valid'))
    model.add(Conv2D(24, (3, 3), strides=1, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(24, (3, 3), strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(5,5), strides=5, padding='valid'))
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def fr_deep_real_same(random_state,input_shape,num_classes):
    # Transfer learning for radio galaxy classification
    # Primary Author: Hongming Tang
    # https://github.com/HongmingTang060313/FR-DEEP
    # https://academic.oup.com/mnras/article/488/3/3358/5538844
    # https://arxiv.org/pdf/1903.11921.pdf
    # Architecture description: Page 7
    # 25 July 2019
    # Notes: This is the original architecture
    
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(11,11),strides=1, padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='same'))
    model.add(Conv2D(16, (5, 5),strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=3, padding='same'))
    model.add(Conv2D(24, (3, 3), strides=1, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(24, (3, 3), strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(5,5), strides=5, padding='same'))
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def fr_deep_erf(random_state,input_shape,num_classes):
    # Transfer learning for radio galaxy classification
    # Primary Author: Hongming Tang
    # https://github.com/HongmingTang060313/FR-DEEP
    # https://academic.oup.com/mnras/article/488/3/3358/5538844
    # https://arxiv.org/pdf/1903.11921.pdf
    # Architecture description: Page 7
    # 25 July 2019
    # Notes: This is the original architecture
    
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(11,11),strides=1, padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),strides=2, padding='valid'))
    model.add(Conv2D(16, (5, 5),strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=3, padding='valid'))
    model.add(Conv2D(24, (3, 3), strides=1, padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(24, (3, 3), strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), strides=1,padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(5,5), strides=5, padding='valid', name='final_output'))
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state), bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def hosenie(random_state,input_shape,num_classes):
    # Source classification in deep radio surveys using machine learning techniques
    # Primary Author: Zafiirah Banon Hosenie
    # http://hdl.handle.net/10394/31250
    # Architecture description: Page 97
    # Date of thesis hand-in: 2017-11-20
    
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(11,11), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(19, kernel_size=(5,5), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(38, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(26, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(26, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),name='final_output'))
    model.add(Flatten())
    model.add(Dense(40, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dense(40, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Activation('softmax'))
    return model

def hosenie_no_bn(random_state,input_shape,num_classes):
    # Source classification in deep radio surveys using machine learning techniques
    # Primary Author: Zafiirah Banon Hosenie
    # http://hdl.handle.net/10394/31250
    # Architecture description: Page 97
    # Date of thesis hand-in: 2017-11-20
    
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(11,11), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(19, kernel_size=(5,5), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(38, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(26, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(26, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),name='final_output'))
    model.add(Flatten())
    model.add(Dense(40, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dense(40, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Activation('softmax'))
    return model

def hosenie_dropout(random_state,input_shape,num_classes):
    # Source classification in deep radio surveys using machine learning techniques
    # Primary Author: Zafiirah Banon Hosenie
    # http://hdl.handle.net/10394/31250
    # Architecture description: Page 97
    # Date of thesis hand-in: 2017-11-20
    
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(11,11), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(19, kernel_size=(5,5), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(38, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(26, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(26, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),name='final_output'))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Flatten())
    model.add(Dense(40, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(40, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Activation('softmax'))
    return model

def hosenie_erf(random_state,input_shape,num_classes):
    # Source classification in deep radio surveys using machine learning techniques
    # Primary Author: Zafiirah Banon Hosenie
    # http://hdl.handle.net/10394/31250
    # Architecture description: Page 97
    
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(11,11), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(19, kernel_size=(5,5), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(38, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(26, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(26, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),name='final_output'))
    model.add(Flatten())
    model.add(Dense(40, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dense(40, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Activation('softmax'))
    return model

def convosource(random_state,input_shape,num_classes):
    # Title: ConvoSource: Radio-Astronomical Source-Finding with Convolutional Neural Networks
    # Author: Vesna Lukic
    # https://arxiv.org/pdf/1910.03631.pdf

    # https://github.com/vlukic973/ConvoSource/blob/master/source_finding_DNN_Bx_yh_v3.py
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(7,7), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(32, (5, 5), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='final_output'))
    # model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def convosource_erf(random_state,input_shape,num_classes):
    # Title: ConvoSource: Radio-Astronomical Source-Finding with Convolutional Neural Networks
    # Author: Vesna Lukic
    # https://arxiv.org/pdf/1910.03631.pdf

    # https://github.com/vlukic973/ConvoSource/blob/master/source_finding_DNN_Bx_yh_v3.py
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(7,7), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    # model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(32, (5, 5), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='final_output'))
    # model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def deepsource(random_state,input_shape,num_classes):
    # Title: DeepSource: point source detection using deep learning
    # Author: A Vafaei Sadr
    # https://academic.oup.com/mnras/article-abstract/484/2/2793/5292502?redirectedFrom=fulltext
    #

    # https://github.com/vafaei-ar/deepsource/blob/master/deepsource/networks.py
        
    input1 = Input(shape=input_shape)
    x1 = Conv2D(16,activation='relu', kernel_size=(5,5),padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')(input1)
    x2 = Conv2D(16,activation='relu', kernel_size=(5,5),padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')(x1)
    x3 = Conv2D(16,activation='relu', kernel_size=(5,5),padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')(x1)
    added = Add()([x1,x3])
    bn = BatchNormalization()(added)
    x4 = Conv2D(16,activation='relu', kernel_size=(5,5),padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')(bn)
    drop = Dropout(0.5, seed=random_state)(x4)
    x5 = Conv2D(1,activation='relu', kernel_size=(5,5),padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')(drop)
    flat = Flatten()(x5)
    dense1 = Dense(500, activation = 'relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')(flat)
    output1 = Dense(num_classes, activation = 'softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')(dense1)
    model = Model(inputs=input1,outputs=output1)
    return model

def first_class(random_state,input_shape,num_classes):
    # Original
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(194, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2,name='final_output'))
    model.add(Flatten())
    model.add(Dense(194, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def first_class_do(random_state,input_shape,num_classes):
    # FirstClass with Dropout, no L2 reg
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(194, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2,name='final_output'))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Flatten())
    model.add(Dense(194, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def first_class_regularized(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(194, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2,name='final_output'))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Flatten())
    model.add(Dense(194, kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def first_class_linear_no_do(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    # model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(194, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2,name='final_output'))
    # model.add(Dropout(0.25,seed=random_state))
    model.add(Flatten())
    model.add(Dense(194, kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('linear'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def first_class_l2_no_do(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    # model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    # model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(194, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2,name='final_output'))
    # model.add(Dropout(0.25,seed=random_state))
    model.add(Flatten())
    model.add(Dense(194, kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def first_class_linear(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Conv2D(194, (3, 3), padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2,name='final_output'))
    model.add(Dropout(0.25,seed=random_state))
    model.add(Flatten())
    model.add(Dense(194, kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('linear'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('softmax'))
    return model

def atlas(random_state,input_shape,num_classes):
    # Paper Title: Radio Galaxy Zoo: Machine learning for radio source host galaxycross-identification
    # Primary Author: M.J. Alger
    # https://doi.org/10.1093/mnras/sty1308
    # https://arxiv.org/pdf/1805.05540.pdf
    # 18 May 2018
    # Repo: https://github.com/chengsoonong/crowdastro/blob/master/notebooks/14_cnn.ipynb

    #Notes: The activation function in the output layer was changed from sigmoid to softmax
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(10, 10),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(MaxPooling2D((5, 5),padding='same'))    
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (10, 10), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_state))
    # This layer's activation was changed from sigmoid to softmax
    model.add(Dense(num_classes, activation='sigmoid', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def atlas_softmax(random_state,input_shape,num_classes):
    # Paper Title: Radio Galaxy Zoo: Machine learning for radio source host galaxycross-identification
    # Primary Author: M.J. Alger
    # https://doi.org/10.1093/mnras/sty1308
    # https://arxiv.org/pdf/1805.05540.pdf
    # 18 May 2018
    # Repo: https://github.com/chengsoonong/crowdastro/blob/master/notebooks/14_cnn.ipynb

    #Notes: The activation function in the output layer was changed from sigmoid to softmax
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(10, 10),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(MaxPooling2D((5, 5),padding='same'))    
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (10, 10), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_state))
    # This layer's activation was changed from sigmoid to softmax
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def atlas_no_do(random_state,input_shape,num_classes):
    # Paper Title: Radio Galaxy Zoo: Machine learning for radio source host galaxycross-identification
    # Primary Author: M.J. Alger
    # https://doi.org/10.1093/mnras/sty1308
    # https://arxiv.org/pdf/1805.05540.pdf
    # 18 May 2018
    # Repo: https://github.com/chengsoonong/crowdastro/blob/master/notebooks/14_cnn.ipynb

    #Notes: The activation function in the output layer was changed from sigmoid to softmax
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(10, 10),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(MaxPooling2D((5, 5),padding='same'))    
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (10, 10), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
    # model.add(Dropout(0.5, seed=random_state))
    # This layer's activation was changed from sigmoid to softmax
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def atlas_no_conv_do(random_state,input_shape,num_classes):
    # Paper Title: Radio Galaxy Zoo: Machine learning for radio source host galaxycross-identification
    # Primary Author: M.J. Alger
    # https://doi.org/10.1093/mnras/sty1308
    # https://arxiv.org/pdf/1805.05540.pdf
    # 18 May 2018
    # Repo: https://github.com/chengsoonong/crowdastro/blob/master/notebooks/14_cnn.ipynb

    #Notes: The activation function in the output layer was changed from sigmoid to softmax
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(10, 10),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(MaxPooling2D((5, 5),padding='same'))    
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (10, 10), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_state))
    # This layer's activation was changed from sigmoid to softmax
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def atlas_linear_no_do(random_state,input_shape,num_classes):
    # Paper Title: Radio Galaxy Zoo: Machine learning for radio source host galaxycross-identification
    # Primary Author: M.J. Alger
    # https://doi.org/10.1093/mnras/sty1308
    # https://arxiv.org/pdf/1805.05540.pdf
    # 18 May 2018
    # Repo: https://github.com/chengsoonong/crowdastro/blob/master/notebooks/14_cnn.ipynb

    #Notes: The activation function in the output layer was changed from sigmoid to softmax
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(10, 10),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(MaxPooling2D((5, 5),padding='same'))    
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (10, 10), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_state))
    # This layer's activation was changed from sigmoid to softmax
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def atlas_l2_no_do(random_state,input_shape,num_classes):
    # Paper Title: Radio Galaxy Zoo: Machine learning for radio source host galaxycross-identification
    # Primary Author: M.J. Alger
    # https://doi.org/10.1093/mnras/sty1308
    # https://arxiv.org/pdf/1805.05540.pdf
    # 18 May 2018
    # Repo: https://github.com/chengsoonong/crowdastro/blob/master/notebooks/14_cnn.ipynb

    #Notes: The activation function in the output layer was changed from sigmoid to softmax
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(10, 10),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(MaxPooling2D((5, 5),padding='same'))    
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (10, 10), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_state))
    # This layer's activation was changed from sigmoid to softmax
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def atlas_l2_do(random_state,input_shape,num_classes):
    # Paper Title: Radio Galaxy Zoo: Machine learning for radio source host galaxycross-identification
    # Primary Author: M.J. Alger
    # https://doi.org/10.1093/mnras/sty1308
    # https://arxiv.org/pdf/1805.05540.pdf
    # 18 May 2018
    # Repo: https://github.com/chengsoonong/crowdastro/blob/master/notebooks/14_cnn.ipynb

    #Notes: The activation function in the output layer was changed from sigmoid to softmax
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(10, 10),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(MaxPooling2D((5, 5),padding='same'))    
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (10, 10), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_state))
    # This layer's activation was changed from sigmoid to softmax
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def atlas_softmax_linear(random_state,input_shape,num_classes):
    # Paper Title: Radio Galaxy Zoo: Machine learning for radio source host galaxycross-identification
    # Primary Author: M.J. Alger
    # https://doi.org/10.1093/mnras/sty1308
    # https://arxiv.org/pdf/1805.05540.pdf
    # 18 May 2018
    # Repo: https://github.com/chengsoonong/crowdastro/blob/master/notebooks/14_cnn.ipynb

    #Notes: The activation function in the output layer was changed from sigmoid to softmax
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(10, 10),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(MaxPooling2D((5, 5),padding='same'))    
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (10, 10), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_state))
    # This layer's activation was changed from sigmoid to softmax
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def atlas_erf(random_state,input_shape,num_classes):
    # Paper Title: Radio Galaxy Zoo: Machine learning for radio source host galaxycross-identification
    # Primary Author: M.J. Alger
    # https://doi.org/10.1093/mnras/sty1308
    # https://arxiv.org/pdf/1805.05540.pdf
    # 18 May 2018
    # Repo: https://github.com/chengsoonong/crowdastro/blob/master/notebooks/14_cnn.ipynb

    #Notes: The activation function in the output layer was changed from sigmoid to softmax
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(10, 10),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(MaxPooling2D((5, 5),padding='same'))    
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(32, (10, 10), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_state))
    # This layer's activation was changed from sigmoid to softmax
    model.add(Dense(num_classes, activation='sigmoid', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

# def atlas_relu(random_state,input_shape,num_classes):
#     # Paper Title: Radio Galaxy Zoo: Machine learning for radio source host galaxycross-identification
#     # Primary Author: M.J. Alger
#     # https://doi.org/10.1093/mnras/sty1308
#     # https://arxiv.org/pdf/1805.05540.pdf
#     # 18 May 2018
#     # Repo: https://github.com/chengsoonong/crowdastro/blob/master/notebooks/14_cnn.ipynb

#     #Notes: The activation function in the output layer was changed from sigmoid to softmax
#     # Dense layer activation was changed to relu
#     
#     model = Sequential()
#     model.add(Conv2D(32, kernel_size=(10, 10),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
#     model.add(MaxPooling2D((5, 5),padding='same'))    
#     model.add(Dropout(0.25, seed=random_state))
#     model.add(Conv2D(32, (10, 10), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
#     model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
#     model.add(Dropout(0.25, seed=random_state))
#     model.add(Flatten())
#     model.add(Dense(64, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
#     model.add(Dropout(0.5, seed=random_state))
#     # This layer's activation was changed from sigmoid to softmax
#     model.add(Dense(num_classes, activation='softmax'))
#     return model

# def atlas_linear(random_state,input_shape,num_classes):
#     # Paper Title: Radio Galaxy Zoo: Machine learning for radio source host galaxycross-identification
#     # Primary Author: M.J. Alger
#     # https://doi.org/10.1093/mnras/sty1308
#     # https://arxiv.org/pdf/1805.05540.pdf
#     # 18 May 2018
#     # Repo: https://github.com/chengsoonong/crowdastro/blob/master/notebooks/14_cnn.ipynb

#     #Notes: The activation function in the output layer was changed from sigmoid to softmax
#     # Dense layer activation was changed to linear
#     
#     model = Sequential()
#     model.add(Conv2D(32, kernel_size=(10, 10),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
#     model.add(MaxPooling2D((5, 5),padding='same'))    
#     model.add(Dropout(0.25, seed=random_state))
#     model.add(Conv2D(32, (10, 10), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
#     model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
#     model.add(Dropout(0.25, seed=random_state))
#     model.add(Flatten())
#     model.add(Dense(64, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
#     model.add(Dropout(0.5, seed=random_state))
#     # This layer's activation was changed from sigmoid to softmax
#     model.add(Dense(num_classes, activation='softmax'))
#     return model

# Our own models

def large_net(random_state,input_shape,num_classes):
    # Based on the ConvNet-4 architecture
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (5, 5),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(MaxPooling2D((2, 2),padding='same'))    
    model.add(Conv2D(64, (5, 5), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (5, 5), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
    model.add(Dropout(0.5, seed=random_state))   
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

# def conv8_bn(random_state,input_shape,num_classes):
#     
#     model = Sequential()
#     model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
#     model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D((2, 2),padding='same'))
#     model.add(Dropout(0.25, seed=random_state))
#     model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
#     model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
#     model.add(Dropout(0.25, seed=random_state))
#     model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
#     model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
#     model.add(Dropout(0.25, seed=random_state))
#     model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
#     model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
#     model.add(Dropout(0.25, seed=random_state))
#     model.add(Flatten())
#     model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
#     model.add(Dropout(0.5, seed=random_state))
#     model.add(Dense(num_classes, activation='softmax'))
#     return model

def conv_deep(random_state,input_shape,num_classes):
    # Conv layers of ConvNet8 and the Dense layers of FR-Deep
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def deepconvX(random_state,input_shape,num_classes):
    # The Conv layer of ConvX with the Dense layers of FR-Deep
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(256, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convX(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convX_erf(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convX_strides(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),strides=2,activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),strides=2,activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),strides=2,activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convX_strides_erf(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),strides=2,activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),strides=2,activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),strides=2,activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convX_strides_less(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),strides=2,activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convX_strides_less_erf(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),strides=2,activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convXpress(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),strides=2,activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convXpress_l2_do(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),strides=2,activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convXpress_l2_no_do(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),strides=2,activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convXpress_l2_full(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),strides=2,activation='relu',padding='same',input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convXpress_leaky(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),strides=2,activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3),  activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3),  activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3),  activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3),  activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3),  activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3),  activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3),  activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500,kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Activation(tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convXtreme(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),strides=2,activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),strides=2,padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    # model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convX_strides_even_less_erf(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),strides=2,activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convX_strides_slightly_less(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),strides=2,activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convX_strides_faster(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),strides=3,activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3),strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model


def convX_strides_close_rf(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),strides=2,activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3),strides=1, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), strides=2, activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convX_strides_first_layer(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),strides=2,activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),strides=1,activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),strides=2,activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convX_stride_first(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),strides=2,activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),strides=1,activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),strides=1,activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convX_mp_valid(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convXrelu(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='relu', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def convXplore(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    # model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def conv25X(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def conv25X_erf(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def conv25X_valid(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def conv25X_valid_erf(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',name='final_output'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def conv2X_final(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def conv3X_final_xplore(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(32, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))#, kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def vgg19(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='valid',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3),activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dense(4096, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dense(1000, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def vgg16_d_valid(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='valid',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3),activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='valid', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def vgg16_d(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def vgg16_d_real(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(64, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='valid',strides=2))
    

    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    

    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    

    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2,name='final_output'))
    

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def vgg16_d_l2_no_do(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(64, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='valid',strides=2))
    

    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    

    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    

    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2,name='final_output'))
    

    model.add(Flatten())
    model.add(Dense(4096, activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def vgg16_d_l2_do(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(64, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='valid',strides=2))
    model.add(Dropout(0.25, seed=random_state))

    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    model.add(Dropout(0.25, seed=random_state))

    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    model.add(Dropout(0.25, seed=random_state))

    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2,name='final_output'))
    model.add(Dropout(0.25, seed=random_state))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def vgg16_d_real_linear(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(64, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='valid',strides=2))
    

    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    

    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    

    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2,name='final_output'))
    

    model.add(Flatten())
    model.add(Dense(4096, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def vgg16_d_real_less_regularized(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(64, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='valid',strides=2))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    # model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2,name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def vgg16_d_real_regularized(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(64, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='valid',strides=2))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid',strides=2,name='final_output'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def vgg16_d_real_same(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros',name='conv2d_input'))
    model.add(Conv2D(64, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='same',strides=2))
    

    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',strides=2))
    

    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',strides=2))
    

    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',strides=2))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same',strides=2,name='final_output'))
    

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(4096, activation='relu', kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def vgg16_d_mod(random_state,input_shape,num_classes):
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(64, (3, 3),activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D((2, 2),padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(Dropout(0.25, seed=random_state))
    model.add(Flatten())
    model.add(Dense(500, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    model.add(Dropout(0.5, seed=random_state))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros'))
    return model

def maslej(random_state,input_shape,num_classes):
    #https://academic.oup.com/mnras/article-abstract/505/1/1464/6276747?redirectedFrom=fulltext
    #https://github.com/VieraMaslej/RadioGalaxy
    
    inputs = Input(shape=input_shape)
    a = tf.keras.layers.Conv2D(128, kernel_size = (3,3), padding = "valid", strides=(2, 2), activation='relu',kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')(inputs)
    a = tf.keras.layers.Flatten()(a)

    b = tf.keras.layers.Conv2D(64, kernel_size = (4,4), padding = "valid", activation='relu',kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')(inputs)
    b = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(b)
    b = tf.keras.layers.Flatten()(b)

    c = tf.keras.layers.Conv2D(64, kernel_size = (2,2), padding = "valid", activation='relu',kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')(inputs)
    c = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(c)
    c = tf.keras.layers.Flatten()(c)
    
    x = tf.keras.layers.concatenate([a, b, c])
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu',kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')(x)
    output = tf.keras.layers.Dense(units = num_classes, activation='softmax',kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model


def maslej_l2_no_do(random_state,input_shape,num_classes):
    #https://academic.oup.com/mnras/article-abstract/505/1/1464/6276747?redirectedFrom=fulltext
    #https://github.com/VieraMaslej/RadioGalaxy
    
    inputs = Input(shape=input_shape)
    a = tf.keras.layers.Conv2D(128, kernel_size = (3,3), padding = "valid", strides=(2, 2), activation='relu',kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')(inputs)
    a = tf.keras.layers.Flatten()(a)

    b = tf.keras.layers.Conv2D(64, kernel_size = (4,4), padding = "valid", activation='relu',kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')(inputs)
    b = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(b)
    b = tf.keras.layers.Flatten()(b)

    c = tf.keras.layers.Conv2D(64, kernel_size = (2,2), padding = "valid", activation='relu',kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')(inputs)
    c = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(c)
    c = tf.keras.layers.Flatten()(c)
    
    x = tf.keras.layers.concatenate([a, b, c])
    # x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01),kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(units = num_classes, activation='softmax',kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model

def maslej_l2_do(random_state,input_shape,num_classes):
    #https://academic.oup.com/mnras/article-abstract/505/1/1464/6276747?redirectedFrom=fulltext
    #https://github.com/VieraMaslej/RadioGalaxy
    
    inputs = Input(shape=input_shape)
    a = tf.keras.layers.Conv2D(128, kernel_size = (3,3), padding = "valid", strides=(2, 2), activation='relu',kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')(inputs)
    a = tf.keras.layers.Dropout(0.5)(a)
    a = tf.keras.layers.Flatten()(a)

    b = tf.keras.layers.Conv2D(64, kernel_size = (4,4), padding = "valid", activation='relu',kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')(inputs)
    b = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(b)
    b= tf.keras.layers.Dropout(0.5)(b)
    b = tf.keras.layers.Flatten()(b)

    c = tf.keras.layers.Conv2D(64, kernel_size = (2,2), padding = "valid", activation='relu',kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros')(inputs)
    c = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(c)
    c = tf.keras.layers.Dropout(0.5)(c)
    c = tf.keras.layers.Flatten()(c)
    
    x = tf.keras.layers.concatenate([a, b, c])
    # x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01),kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(units = num_classes, activation='softmax',kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model


# Architecture Template
# Build and test your own network 
# def your_own_net(random_state,input_shape,num_classes):#     
#     
#     model = Sequential()
    # model.add(Conv2D( ?, kernel_size=( ?,  ?),activation='relu',padding='same',input_shape=input_shape, kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
#     model.add(Dropout( ?, seed=random_state))
#     model.add(MaxPooling2D((?, ?),padding='same'))
#     model.add(Flatten())
#     model.add(Dense(?, activation='linear',kernel_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_uniform(seed=random_state),bias_initializer='zeros')) 
#     model.add(Dropout(?, seed=random_state))   
#     model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.he_normal(seed=random_state),bias_initializer='zeros'))
#     return model
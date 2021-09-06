import tensorflow as tf
import numpy as np
import tensorflow.keras.callbacks as Callback
import sys
import os
import random
import model_architectures as mac
from tensorflow.keras import initializers
from sklearn.metrics import confusion_matrix
from skimage.transform import rotate
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from dataloader import DataGenerator
from read_architectures import get_class_label
from astropy.io import fits
import tensorflow.keras.datasets.mnist as mnist
import time

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def get_batch(size):
    factors = factorize(size)
    if len(factors) > 2:
        return 32
    else:
        #prime
        return 32
    return 32

def lr_scheduler(epoch, lr):
    if epoch < 8:
        return lr
    else:
        return lr*tf.math.exp(-0.1)

def test(model_name,architecture, SEED, results_path, data_path, model_path, partition, labels, img_rows, img_cols, final=False):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    LR = float(architecture[0])
    EP = int(architecture[1])
    if tf.keras.backend.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
    label_map = get_class_label()
    num_classes = len(label_map.keys())
    training_generator = DataGenerator(partition["train"], labels, data_path, label_map, SEED, tf.keras.backend.image_data_format(),img_rows=img_rows, img_cols=img_cols, batch_size=64, shuffle=False)
    validation_generator = DataGenerator(partition["validation"], labels, data_path, label_map, SEED, tf.keras.backend.image_data_format(), img_rows=img_rows, img_cols=img_cols, shuffle=False, batch_size=64)
    if final == False:
        model = load_model(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'.h5',compile=False)
    else:
        model = load_model(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'_final.h5',compile=False)
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    lr_metric = get_lr_metric(opt)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt,
        loss=loss_fn,
        metrics=['accuracy',lr_metric])
    train_results = model.evaluate(training_generator)
    val_results = model.evaluate(validation_generator)
    print(train_results)
    print(val_results)
    print(train_results[0]/val_results[0])
    f = open("loss_log_trainvsval2.txt","a")
    f.write(model_name+'_final'+','+str(train_results[0])+','+str(val_results[0])+'\n')
    f.close()
    return train_results, val_results
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

def train_mnist(model_name, architecture, SEED, results_path, data_path, model_path, partition, labels, img_rows, img_cols):
    print("Training "+model_name)
    LR = float(architecture[0])
    EP = int(architecture[1])
    os.environ['PYTHONHASHSEED']=str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    num_classes = 10
    input_shape = (28, 28, 1)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    val_size = int(np.floor(x_train.shape[0]/20))
    print("x_train shape:", x_train.shape)
    print("Validation size: ",val_size)
    x_val = x_train[:val_size,:,:]
    y_val = y_train[:val_size]
    x_train = x_train[val_size:,:,:]
    y_train = y_train[val_size:]
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    x_val = np.expand_dims(x_val, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")
    print(x_val.shape[0], "validation samples")
    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)
    model_call = getattr(mac, model_name)
    model = model_call(SEED,input_shape,num_classes)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=LR,decay_steps=10000,decay_rate=LR/EP)
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    lr_metric = get_lr_metric(opt)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    mcp_save = ModelCheckpoint(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'_MNIST.h5', save_best_only=True, monitor='val_loss', mode='min')
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=1e-10, min_delta=0.001,mode='min')
    model.compile(optimizer=opt,
        loss=loss_fn,
        metrics=['accuracy',lr_metric])
    model.fit(x_train,y_train,
                    validation_data=(x_val,y_val), epochs=EP,callbacks=[mcp_save,reduce_lr],verbose=2)
    model.save(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'_final_MNIST.h5')
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(cm)
    ncm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(ncm)
    mpca = 0
    for i in range(num_classes):
        mpca = ncm[i,i] + mpca
    mpca = mpca/num_classes
    mpca = mpca*100
    print(mpca)
    return cm, ncm, mpca

def test_mnist(model_name, architecture, SEED, results_path, data_path, model_path, partition, labels, img_rows, img_cols, final=False):
    LR = float(architecture[0])
    EP = int(architecture[1])
    os.environ['PYTHONHASHSEED']=str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    num_classes = 10
    input_shape = (28, 28, 1)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    val_size = int(np.floor(x_train.shape[0]/20))
    print("x_train shape:", x_train.shape)
    print("Validation size: ",val_size)
    x_val = x_train[:val_size,:,:]
    y_val = y_train[:val_size]
    x_train = x_train[val_size:,:,:]
    y_train = y_train[val_size:]
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    x_val = np.expand_dims(x_val, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")
    print(x_val.shape[0], "validation samples")
    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)
    if final == False:
        model = load_model(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'_MNIST.h5',compile=False)
    else:
        model = load_model(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'_final_MNIST.h5',compile=False)
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    lr_metric = get_lr_metric(opt)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt,
        loss=loss_fn,
        metrics=['accuracy',lr_metric])
    t0 = time.time()
    y_pred = model.predict(x_test,verbose=2)
    t1 = time.time()
    time_dif = t1 - t0
    print("Time:")
    print(time_dif)
    print("IPS")
    print(len(x_test)/time_dif)
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(cm)
    ncm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(ncm)
    mpca = 0
    for i in range(num_classes):
        mpca = ncm[i,i] + mpca
    mpca = mpca/num_classes
    mpca = mpca*100
    print(mpca)
    return cm, ncm, mpca, time_dif

def train(model_name, architecture, SEED, results_path, data_path, model_path, partition, labels, img_rows, img_cols):
    print("Training "+model_name)
    os.environ['PYTHONHASHSEED']=str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    label_map = get_class_label()
    num_classes = len(label_map.keys())
    training_generator = DataGenerator(partition["train"], labels, data_path, label_map, SEED, tf.keras.backend.image_data_format(),img_rows=img_rows, img_cols=img_cols, batch_size=32, shuffle=False)
    validation_generator = DataGenerator(partition["validation"], labels, data_path, label_map, SEED, tf.keras.backend.image_data_format(), img_rows=img_rows, img_cols=img_cols, shuffle=False, batch_size=32)
    LR = float(architecture[0])
    EP = int(architecture[1])
    input_shape = (img_rows, img_cols, 1)
    model_call = getattr(mac, model_name)
    model = model_call(SEED,input_shape,num_classes)
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    lr_metric = get_lr_metric(opt)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    mcp_save = ModelCheckpoint(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'.h5', save_best_only=True, monitor='val_loss', mode='min')
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=1e-10, min_delta=0.001,mode='min')
    model.compile(optimizer=opt,
        loss=loss_fn,
        metrics=['accuracy',lr_metric])
    model.fit(training_generator, validation_data=validation_generator, epochs=EP,callbacks=[mcp_save,reduce_lr],verbose=2)
    test_generator = DataGenerator(partition["test"], labels, data_path, label_map, SEED, tf.keras.backend.image_data_format(), img_rows=img_rows, img_cols=img_cols,shuffle=False,batch_size=32)
    model.save(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'_final.h5')
    print(model.evaluate(test_generator, verbose=2))
    print()

def test(model_name,architecture, SEED, results_path, data_path, model_path, partition, labels, img_rows, img_cols, final=False):
    LR = float(architecture[0])
    EP = int(architecture[1])
    if tf.keras.backend.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
    label_map = get_class_label()
    num_classes = len(label_map.keys())
    test_generator = DataGenerator(partition["test"], labels, data_path, label_map, SEED, tf.keras.backend.image_data_format(), img_rows=img_rows, img_cols=img_cols,shuffle=False,batch_size=32)
    y_test = np.zeros([len(partition["test"]), num_classes])
    for i, key in enumerate(partition["test"]):
        y_test[i,labels[key]] = 1
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
    t0 = time.time()
    y_pred = model.predict(test_generator,verbose=2)
    t1 = time.time()
    time_dif = t1 - t0
    print("Time:")
    print(time_dif)
    print("IPS:")
    print(len(partition["test"])/time_dif)
    if final:
        np.save(results_path+"/"+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+"_final.npy",y_pred)
    else:
        np.save(results_path+"/"+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+".npy",y_pred)
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(cm)
    ncm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(ncm)
    mpca = 0
    for i in range(num_classes):
        mpca = ncm[i,i] + mpca
    mpca = mpca/num_classes
    mpca = mpca*100
    print(mpca)
    print()
    return cm, ncm, mpca, time_dif

def save_test_labels(SEED, results_path, data_path, model_path, partition, labels, img_rows, img_cols, train_size, val_size, final=False):
    LR = float(architecture[0])
    EP = int(architecture[1])
    if tf.keras.backend.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
    label_map = get_class_label()
    num_classes = len(label_map.keys())
    # test_generator = DataGenerator(partition["test"], labels, data_path, label_map, SEED, tf.keras.backend.image_data_format(), img_rows=img_rows, img_cols=img_cols,shuffle=False,batch_size=32)
    y_test = np.zeros([len(partition["test"]), num_classes])
    for i, key in enumerate(partition["test"]):
        y_test[i,labels[key]] = 1
    np.save(results_path+"/"+"y_test_"+str(SEED)+'_'+str(train_size)+'_'+str(val_size)+".npy", y_test)
    return y_test

def time_test(model_name,architecture, SEED, results_path, data_path, model_path, partition, labels, img_rows, img_cols, final=False):
    print("Timing "+model_name)
    LR = float(architecture[0])
    EP = int(architecture[1])
    if tf.keras.backend.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
    label_map = get_class_label()
    num_classes = len(label_map.keys())
    test_generator = DataGenerator(partition["test"], labels, data_path, label_map, SEED, tf.keras.backend.image_data_format(), img_rows=img_rows, img_cols=img_cols,shuffle=False,batch_size=32)
    y_test = np.zeros([len(partition["test"]), num_classes])
    for i, key in enumerate(partition["test"]):
        y_test[i,labels[key]] = 1
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
    ips = []
    time_dif = []
    for counter in range(5):
        t0 = time.time()
        model.predict(test_generator,verbose=2)
        t1 = time.time()
        time_dif.append(t1 - t0)
        ips.append(len(partition["test"])/(t1-t0))
    print("Time:")
    print(time_dif)
    print("IPS:")
    print(ips)
    return time_dif, ips, np.average(time_dif), np.std(time_dif), np.average(ips), np.std(ips)
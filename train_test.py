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
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from dataloader import DataGenerator
from read_architectures import get_class_label
from astropy.io import fits
import tensorflow.keras.datasets.mnist as mnist
import time
import matplotlib.pyplot as plt
import io
# import loss_callback

# class RealTrainLossCallback(Callback):
#     def __init__(self,a,b):
#         self.mods = "hi"

#     def on_epoch_end(self, epoch, logs=None):
#         val_loss = logs['val_loss']
#         val_acc = logs['val_acc']
#         history = self.model.predict(self.x_train)
#         f = open("loss_log_trainvsval.txt","a")
#         f.write('epoch'+str(epoch))
#         f.write('val_loss,'+str(val_loss)+'\ntraining_loss,'+str( history.history['loss'])+'\n')
#         f.write('val_acc,'+str(val_acc)+'\ntraining_acc,'+str( history.history['acc'])+'\n')
#         f.close()
#         print("End epoch {} of training; real training loss: {}".format(epoch, history.history['loss']))

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

def train_mnist(model_name, architecture, SEED, results_path, data_path, model_path, img_rows, img_cols):
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
                              patience=3, min_lr=1e-12, min_delta=0.001,mode='min')
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
    #Testing
    train_results = model.evaluate(x_train,y_train)
    val_results = model.evaluate(x_val,y_val)
    # Log loss and acc ratios
    f = open(results_path+"/"+model_name+"_final_MNIST.txt","a")
    f.write(model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'\n')
    f.write("train_loss: "+str(train_results[0])+'\n')
    f.write("val_loss: "+str(val_results[0])+'\n')
    f.write("ratio: "+str(train_results[0]/val_results[0])+'\n')   
    f.close()
    # Write to CSV file
    csv_exists = os.path.isfile(results_path+"/"+model_name+"_final_MNIST.csv")
    f = open(results_path+"/"+model_name+"_final.csv","a+")
    metric_prf = ['-P','-R','-F1']
    if csv_exists == False:
        f.write("Architectures,Seed,LearningRate,Epochs,train_acc,val_acc,train_loss,val_loss,loss_ratio,acc_ratio,MPCA,")
        for metric_letter in metric_prf: 
            for gg in range(num_classes):
                f.write(str(gg)+metric_letter+',')
        f.write('\n')
    f.write(model_name+","+str(SEED)+","+str(LR)+","+str(EP)+","+str(train_results[1])+","+str(val_results[1])+","+str(train_results[0])+","+str(val_results[0])+","+str(train_results[0]/val_results[0])+","+str(train_results[1]/val_results[1])+",")
    f.close()
    model = load_model(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'_MNIST.h5',compile=False)
    model.compile(optimizer=opt,
        loss=loss_fn,
        metrics=['accuracy',lr_metric])
    train_results = model.evaluate(x_train,y_train)
    val_results = model.evaluate(x_val,y_val)
    f = open(results_path+"/"+model_name+".txt","a")
    f.write(model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'\n')
    f.write("train_acc: "+str(train_results[1])+'\n')
    f.write("val_acc: "+str(val_results[1])+'\n')
    f.write("train_loss: "+str(train_results[0])+'\n')
    f.write("val_loss: "+str(val_results[0])+'\n')
    f.write("ratio: "+str(train_results[0]/val_results[0])+'\n')   
    f.close()
    # Write to arch CSV file
    csv_exists = os.path.isfile(results_path+"/"+model_name+"_MNIST.csv")
    f = open(results_path+"/"+model_name+"_MNIST.csv","a+")
    if csv_exists == False:
        f.write("Architectures,Seed,LearningRate,Epochs,train_acc,val_acc,train_loss,val_loss,loss_ratio,acc_ratio,MPCA,")
        for metric_letter in metric_prf: 
            for gg in range(num_classes):
                f.write(str(gg)+metric_letter+',')
        f.write('\n')
    f.write(model_name+","+str(SEED)+","+str(LR)+","+str(EP)+","+str(train_results[1])+","+str(val_results[1])+","+str(train_results[0])+","+str(val_results[0])+","+str(train_results[0]/val_results[0])+","+str(train_results[1]/val_results[1])+",")
    f.close()
    return cm, ncm, mpca

def test_mnist(model_name, architecture, SEED, results_path, data_path, model_path, img_rows, img_cols, final=False):
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
    precision_indep = []
    recall_indep = []
    f1score_indep = []
    matrixN = ncm
    for i in range(num_classes):
        precision_sum = 0
        recall_sum = 0
        for j in range(num_classes):
            precision_sum = matrixN[j,i] + precision_sum
            recall_sum = matrixN[i,j] + recall_sum
        precision_indep.append(matrixN[i,i]/precision_sum)
        recall_indep.append(matrixN[i,i]/recall_sum)
        f1score_indep.append((2*precision_indep[i]*recall_indep[i])/(precision_indep[i]+recall_indep[i]))
    if final:
        f = open(results_path+"/"+model_name+"_final_MNIST.csv","a")
    else:
        f = open(results_path+"/"+model_name+"_MNIST.csv","a")        
    f.write(str(mpca)+',')
    for i in precision_indep:
        f.write(str(i)+',')
    for i in recall_indep:
        f.write(str(i)+',')
    for i in f1score_indep:
        f.write(str(i)+',')
    f.write('\n')
    f.close()
    return cm, ncm, mpca, time_dif

def train(model_name, architecture, SEED, results_path, data_path, model_path, partition, labels, img_rows, img_cols, early_stop_epochs):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Training "+model_name)
    print("Seed: "+str(SEED))
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
    decay_rate = 0.1/EP    
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    lr_metric = get_lr_metric(opt)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    mcp_save = ModelCheckpoint(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'.h5', save_best_only=True, monitor='val_loss', mode='min')
    # lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=1e-12, min_delta=0.01,mode='min')
    callbacks_list = [mcp_save,reduce_lr]
    if early_stop_epochs != -1:
        print("Early stopping on, setting patience to "+str(early_stop_epochs)+" epochs")
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop_epochs, min_delta=0.01)
        callbacks_list.append(early_stopping)
    else:
        print("Early stopping off")
    model.compile(optimizer=opt,
        loss=loss_fn,
        metrics=['accuracy',lr_metric])
    f = open("loss_log_trainvsval.txt","a")
    f.write('Model_name,'+model_name+'\n')
    f.close()
    history = model.fit(training_generator, validation_data=validation_generator, epochs=EP,callbacks=callbacks_list,verbose=2)
    # test_generator = DataGenerator(partition["test"], labels, data_path, label_map, SEED, tf.keras.backend.image_data_format(), img_rows=img_rows, img_cols=img_cols,shuffle=False,batch_size=32)
    model.save(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'_final.h5')
    # print(model.evaluate(test_generator, verbose=2))
    print("Losses")
    print(history.history['loss'])
    print(history.history['val_loss'])
    plt.plot(history.history['loss'],color='b')
    plt.plot(history.history['val_loss'],color='r')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    os.system("mkdir -p loss_images_updated_learning")
    plt.savefig("loss_images_updated_learning/"+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+"_loss.png")
    plt.show()
    plt.clf()
    plt.plot(history.history['accuracy'],color='b')
    plt.plot(history.history['val_accuracy'],color='r')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("loss_images_updated_learning/"+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+"_acc.png")
    plt.show()
    plt.clf()
    # Getting a better value for loss without dropout
    train_results = model.evaluate(training_generator)
    val_results = model.evaluate(validation_generator)
    # Log loss and acc ratios
    f = open(results_path+"/"+model_name+"_final.txt","a")
    f.write(model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'\n')
    f.write("train_loss: "+str(train_results[0])+'\n')
    f.write("val_loss: "+str(val_results[0])+'\n')
    f.write("ratio: "+str(train_results[0]/val_results[0])+'\n')   
    f.close()
    # Write to CSV file
    csv_exists = os.path.isfile(results_path+"/"+model_name+"_final.csv")
    f = open(results_path+"/"+model_name+"_final.csv","a+")
    if csv_exists == False:
        f.write("Architectures,Seed,LearningRate,Epochs,train_acc,val_acc,train_loss,val_loss,loss_ratio,acc_ratio,MPCA,Comp-P,FRI-P,FRII-P,Bent-P,Comp-R,FRI-R,FRII-R,Bent-R,Comp-F1,FRI-F1,FRII-F1,Bent-F1,\n")
    f.write(model_name+","+str(SEED)+","+str(LR)+","+str(EP)+","+str(train_results[1])+","+str(val_results[1])+","+str(train_results[0])+","+str(val_results[0])+","+str(train_results[0]/val_results[0])+","+str(train_results[1]/val_results[1])+",")
    f.close()
    model = load_model(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'.h5',compile=False)
    model.compile(optimizer=opt,
        loss=loss_fn,
        metrics=['accuracy',lr_metric])
    train_results = model.evaluate(training_generator)
    val_results = model.evaluate(validation_generator)
    f = open(results_path+"/"+model_name+".txt","a")
    f.write(model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'\n')
    f.write("train_acc: "+str(train_results[1])+'\n')
    f.write("val_acc: "+str(val_results[1])+'\n')
    f.write("train_loss: "+str(train_results[0])+'\n')
    f.write("val_loss: "+str(val_results[0])+'\n')
    f.write("ratio: "+str(train_results[0]/val_results[0])+'\n')   
    f.close()
    # Write to arch CSV file
    csv_exists = os.path.isfile(results_path+"/"+model_name+".csv")
    f = open(results_path+"/"+model_name+".csv","a+")
    if csv_exists == False:
        f.write("Architectures,Seed,LearningRate,Epochs,train_acc,val_acc,train_loss,val_loss,loss_ratio,acc_ratio,MPCA,Comp-P,FRI-P,FRII-P,Bent-P,Comp-R,FRI-R,FRII-R,Bent-R,Comp-F1,FRI-F1,FRII-F1,Bent-F1,\n")
    f.write(model_name+","+str(SEED)+","+str(LR)+","+str(EP)+","+str(train_results[1])+","+str(val_results[1])+","+str(train_results[0])+","+str(val_results[0])+","+str(train_results[0]/val_results[0])+","+str(train_results[1]/val_results[1])+",")
    f.close()

def validation(model_name,architecture, SEED, results_path, data_path, model_path, partition, labels, img_rows, img_cols):
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
    LR = float(architecture[0])
    EP = int(architecture[1])
    input_shape = (img_rows, img_cols, 1)
    model_call = getattr(mac, model_name)
    # model = model_call(SEED,input_shape,num_classes)    
    decay_rate = 0.1/EP    
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    lr_metric = get_lr_metric(opt)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    mcp_save = ModelCheckpoint(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'.h5', save_best_only=True, monitor='val_loss', mode='min')
    # lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=1e-12, min_delta=0.01,mode='min')
    callbacks_list = [mcp_save,reduce_lr]
    model = load_model(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'_final.h5',compile=False)
    model.compile(optimizer=opt,
        loss=loss_fn,
        metrics=['accuracy',lr_metric])
    # Getting a better value for loss without dropout
    train_results = model.evaluate(training_generator)
    val_results = model.evaluate(validation_generator)
    # Log loss and acc ratios
    f = open(results_path+"/"+model_name+"_final.txt","a")
    f.write(model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'\n')
    f.write("train_loss: "+str(train_results[0])+'\n')
    f.write("val_loss: "+str(val_results[0])+'\n')
    f.write("ratio: "+str(train_results[0]/val_results[0])+'\n')   
    f.close()
    # Write to CSV file
    csv_exists = os.path.isfile(results_path+"/"+model_name+"_final.csv")
    f = open(results_path+"/"+model_name+"_final.csv","a+")
    if csv_exists == False:
        f.write("Architectures,Seed,LearningRate,Epochs,train_acc,val_acc,train_loss,val_loss,loss_ratio,acc_ratio,MPCA,Comp-P,FRI-P,FRII-P,Bent-P,Comp-R,FRI-R,FRII-R,Bent-R,Comp-F1,FRI-F1,FRII-F1,Bent-F1,\n")
    f.write(model_name+","+str(SEED)+","+str(LR)+","+str(EP)+","+str(train_results[1])+","+str(val_results[1])+","+str(train_results[0])+","+str(val_results[0])+","+str(train_results[0]/val_results[0])+","+str(train_results[1]/val_results[1])+",")
    f.close()
    model = load_model(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'.h5',compile=False)
    model.compile(optimizer=opt,
        loss=loss_fn,
        metrics=['accuracy',lr_metric])
    train_results = model.evaluate(training_generator)
    val_results = model.evaluate(validation_generator)
    f = open(results_path+"/"+model_name+".txt","a")
    f.write(model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'\n')
    f.write("train_acc: "+str(train_results[1])+'\n')
    f.write("val_acc: "+str(val_results[1])+'\n')
    f.write("train_loss: "+str(train_results[0])+'\n')
    f.write("val_loss: "+str(val_results[0])+'\n')
    f.write("ratio: "+str(train_results[0]/val_results[0])+'\n')   
    f.close()
    # Write to arch CSV file
    csv_exists = os.path.isfile(results_path+"/"+model_name+".csv")
    f = open(results_path+"/"+model_name+".csv","a+")
    if csv_exists == False:
        f.write("Architectures,Seed,LearningRate,Epochs,train_acc,val_acc,train_loss,val_loss,loss_ratio,acc_ratio,MPCA,Comp-P,FRI-P,FRII-P,Bent-P,Comp-R,FRI-R,FRII-R,Bent-R,Comp-F1,FRI-F1,FRII-F1,Bent-F1,\n")
    f.write(model_name+","+str(SEED)+","+str(LR)+","+str(EP)+","+str(train_results[1])+","+str(val_results[1])+","+str(train_results[0])+","+str(val_results[0])+","+str(train_results[0]/val_results[0])+","+str(train_results[1]/val_results[1])+",")
    f.close()

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
    if final:
        f = open(results_path+"/"+model_name+"_final.txt","a")
    else:
        f = open(results_path+"/"+model_name+".txt","a")        
    # f.write(model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'\n')
    matrixN = ncm
    precision_indep = []
    recall_indep = []
    f1score_indep = []
    for i in range(4):        
        precision_indep.append(matrixN[i,i]/(matrixN[0,i]+matrixN[1,i]+matrixN[2,i]+matrixN[3,i]))
        recall_indep.append(matrixN[i,i]/(matrixN[i,0]+matrixN[i,1]+matrixN[i,2]+matrixN[i,3]))
        f1score_indep.append((2*precision_indep[i]*recall_indep[i])/(precision_indep[i]+recall_indep[i]))
    f.write("Seed:"+'\n')
    f.write(str(SEED)+'\n')
    f.write("MPCA:"+'\n')
    f.write(str(mpca)+'\n')
    f.write("NCM:"+'\n')
    f.write(str(ncm)+'\n')
    f.write("CM:"+'\n')
    f.write(str(cm)+'\n')
    f.write("Precision:"+'\n')
    f.write(str(precision_indep)+'\n')
    f.write("Recall:"+'\n')
    f.write(str(recall_indep)+'\n')
    f.write("F1:"+'\n')
    f.write(str(f1score_indep)+'\n')
    f.write("IPS:"+'\n')
    f.write(str(len(partition["test"])/time_dif)+'\n')
    f.close()
    if final:
        f = open(results_path+"/"+model_name+"_final.csv","a")
    else:
        f = open(results_path+"/"+model_name+".csv","a")        
    f.write(str(mpca)+',')
    for i in precision_indep:
        f.write(str(i)+',')
    for i in recall_indep:
        f.write(str(i)+',')
    for i in f1score_indep:
        f.write(str(i)+',')
    f.write('\n')
    f.close()
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

def get_model_summary(model):
    #Source: User: Pasa https://stackoverflow.com/a/53937848
    #Purpose: convert Keras model summary to string
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

def count_params(model_summary_string):
    dense = 0
    conv = 0
    flatten = 0
    split_string = model_summary_string.split('\n')
    for line in split_string:
        parenthesis_split = line.split(')')
        if len(parenthesis_split) == 3:
            if 'Dense' in parenthesis_split[0]:
                dense+= int(parenthesis_split[2])
            elif 'Conv2D' in parenthesis_split[0]:
                conv+= int(parenthesis_split[2])
            elif 'Flatten' in parenthesis_split[0]:
                print(parenthesis_split[1].split(' ')[-1])
                flatten = int(parenthesis_split[1].split(' ')[-1])
        if "Trainable params: " in line:
            trainable_params = line.split(" ")[2]
    return dense, conv, flatten, trainable_params

def model_parameters(model_name,architecture, SEED, model_path, img_rows, img_cols):
    print("Model summary of "+model_name)
    LR = float(architecture[0])
    EP = int(architecture[1])
    if tf.keras.backend.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
    label_map = get_class_label()
    num_classes = len(label_map.keys())
    model = load_model(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'.h5',compile=False)
    model_summary_string = get_model_summary(model)
    print(model_summary_string)
    dense,conv,flatten,trainable_params = count_params(model_summary_string)
    print('Conv: '+str(conv))
    print('Flatten: '+str(flatten))
    print('Dense: '+str(dense))
    print('Total: '+str(dense+conv))
    return dense, conv, flatten,trainable_params

# def get_flops(model_name,architecture, SEED, model_path, img_rows, img_cols):
#     session = tf.compat.v1.Session()
#     graph = tf.compat.v1.get_default_graph()
        

#     with graph.as_default():
#         LR = float(architecture[0])
#         EP = int(architecture[1])
#         with session.as_default():
#             model = load_model(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'.h5',compile=False)

#             run_meta = tf.compat.v1.RunMetadata()
#             opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
#             # We use the Keras session graph in the call to the profiler.
#             flops = tf.compat.v1.profiler.profile(graph=graph,
#                                                   run_meta=run_meta, cmd='op', options=opts)
#             print(flops.total_float_ops)
#             return flops.total_float_ops

def numel(w : list):
    out = 1
    for k in w:
        out *= k
    return out

def compute_input_flops(layer, macs = False):
    return 0
def compute_padding_flops(layer, macs = False):
    return 0
def compute_activation_flops(layer, macs = False):
    return 0
def compute_tfop_flops(layer, macs = False):
    return 0
def compute_add_flops(layer, macs = False):
    return 0

def compute_conv2d_flops(layer, macs = False):
    
#     _, cin, h, w = input_shape
    if layer.data_format == "channels_first":
        _, input_channels, _, _ = layer.input_shape
        _, output_channels, h, w, = layer.output_shape
    elif layer.data_format == "channels_last":
        _, _, _, input_channels = layer.input_shape
        _, h, w, output_channels = layer.output_shape
    
    w_h, w_w =  layer.kernel_size

#     flops = h * w * output_channels * input_channels * w_h * w_w / (stride**2)
    flops = h * w * output_channels * input_channels * w_h * w_w

    
    if not macs:
        flops_bias = numel(layer.output_shape[1:]) if layer.use_bias is not None else 0
        flops = 2 * flops + flops_bias
        
    return int(flops)
    

def compute_fc_flops(layer, macs = False):
    ft_in, ft_out =  layer.input_shape[-1], layer.output_shape[-1]
    flops = ft_in * ft_out
    
    if not macs:
        flops_bias = ft_out if layer.use_bias is not None else 0
        flops = 2 * flops + flops_bias
        
    return int(flops)

def compute_bn2d_flops(layer, macs = False):
    # subtract, divide, gamma, beta
    flops = 2 * numel(layer.input_shape[1:])
    
    if not macs:
        flops *= 2
    
    return int(flops)


def compute_relu_flops(layer, macs = False):
    
    flops = 0
    if not macs:
        flops = numel(layer.input_shape[1:])

    return int(flops)


def compute_maxpool2d_flops(layer, macs = False):

    flops = 0
    if not macs:
        flops = layer.pool_size[0]**2 * numel(layer.output_shape[1:])

    return flops


def compute_pool2d_flops(layer, macs = False):

    flops = 0
    if not macs:
        flops = layer.pool_size[0]**2 * numel(layer.output_shape[1:])

    return flops

def compute_globalavgpool2d_flops(layer, macs = False):

    if layer.data_format == "channels_first":
        _, input_channels, h, w = layer.input_shape
        _, output_channels = layer.output_shape
    elif layer.data_format == "channels_last":
        _, h, w, input_channels = layer.input_shape
        _, output_channels = layer.output_shape

    return h*w

def compute_softmax_flops(layer, macs = False):
    
    nfeatures = numel(layer.input_shape[1:])
    
    total_exp = nfeatures # https://stackoverflow.com/questions/3979942/what-is-the-complexity-real-cost-of-exp-in-cmath-compared-to-a-flop
    total_add = nfeatures - 1
    total_div = nfeatures
    
    flops = total_div + total_exp
    
    if not macs:
        flops += total_add
        
    return flops

def dropout_flops(layer,macs):
    return 0

def flatten_flops(layer,macs):
    return 0

def save_flops(layer, flops, conv_flops, dense_flops, macs=False):
    get_flops_dict = {'ReLU': compute_relu_flops,
                'InputLayer': compute_input_flops,
                'Conv2D': compute_conv2d_flops,
                'ZeroPadding2D': compute_padding_flops,
                'Activation': compute_activation_flops,
                'Dense': compute_fc_flops,
                'BatchNormalization': compute_bn2d_flops,
                'TensorFlowOpLayer': compute_tfop_flops,
                'MaxPooling2D': compute_pool2d_flops,
                'Add': compute_add_flops,
                'GlobalAveragePooling2D': compute_globalavgpool2d_flops,
                'Dropout':dropout_flops,
                'Flatten':flatten_flops}
    flops_layer = get_flops_dict[layer.__class__.__name__](layer, macs)
    # if macs:
    #     macs.append(flops)
    # else:
    flops.append(flops_layer)       
    return flops
        

def get_stats(model, flops = False, macs = False):
    
    # if params:
    #     self.count_params()
   
    if flops:
        flops_arr = []
        conv_flops = []
        dense_flops = []
    
    # if macs:
    #     macs = []

    if flops:
        for layer in model.layers:
            flops_arr = save_flops(layer, flops_arr,conv_flops, dense_flops)
    # if macs:
    #     for layer in model.layers:
    #         save_flops(layer, macs=True)
    total = sum(flops_arr)
    return flops_arr, total

def get_flops(model_name,architecture, SEED, model_path, img_rows, img_cols):
    LR = float(architecture[0])
    EP = int(architecture[1])
    model = load_model(model_path+model_name+'_'+str(SEED)+'_'+str(LR)+'_'+str(EP)+'.h5',compile=False)

    

    flops, total = get_stats(model, flops = True, macs = False)
    print(flops, total)
    #insert flopco call here
    return total

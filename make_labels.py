import sys
from downloads import download_unLRG, download_LRG
from read_architectures import read_architectures, get_class_label
from results import generate_figures
from make_splits_no_check import make_splits
# from train_test import save_test_labels
import numpy as np
import json
import os.path

def save_test_labels(SEED, results_path, data_path, model_path, partition, labels, img_rows, img_cols, train_size, val_size, final=False):
    input_shape = (img_rows, img_cols, 1)
    label_map = get_class_label()
    num_classes = len(label_map.keys())
    # test_generator = DataGenerator(partition["test"], labels, data_path, label_map, SEED, tf.keras.backend.image_data_format(), img_rows=img_rows, img_cols=img_cols,shuffle=False,batch_size=32)
    y_test = np.zeros([len(partition["test"]), num_classes])
    for i, key in enumerate(partition["test"]):
        y_test[i,labels[key]] = 1
    np.save(results_path+"/"+"y_test_"+str(SEED)+'_'+str(train_size)+'_'+str(val_size)+".npy", y_test)
    return y_test

def main(argv):
    MNIST_test = False
    only_test = "False"
    if len(argv) == 1:
        print("No arguments given, default random seed (8901), rotation factor (15 deg) and architectures file will be used.")
        rotate_factor = 15
        seed = 1826
        architectures_file = "architectures.txt"
        results_path = "rg_class_experiment_results/"
        data_path = "FITS_300/"
        model_path = "models/"
        train_size = 250
        val_size = 100
        img_rows = 300
        img_cols = 300
        # Manifest is a list of files and their classes
        data_manifest = "unLRG_manifest.csv"
    elif len(argv) == 11:
        seed = int(argv[1])
        rotate_factor = int(argv[2])
        architectures_file = argv[3]
        results_path = argv[4]
        data_path = argv[5]
        model_path = argv[6]
        data_manifest = argv[7]
        train_size = int(argv[8])
        val_size = int(argv[9])
        img_rows = int(argv[10])
        img_cols = img_rows
    elif len(argv) == 2:
        if argv[1] == "mnist":
            MNIST_test = True
            img_rows = 28
            img_cols = 28
        elif argv[1] == "test":
            only_test = "default"
            rotate_factor = 15
            seed = 8901
            architectures_file = "architectures.txt"
            results_path = "rg_class_experiment_results/"
            data_path = "FITS_300/"
            model_path = "models/"
            train_size = 250
            val_size = 100
            img_rows = 300
            img_cols = 300
            # Manifest is a list of files and their classes
            data_manifest = "unLRG_manifest.csv"
    elif len(argv) == 3:
        rotate_factor = 15
        architectures_file = "architectures.txt"
        results_path = "rg_class_experiment_results/"
        data_path = "FITS_300/"
        model_path = "models/"
        train_size = 250
        val_size = 100
        img_rows = 300
        img_cols = 300
        data_manifest = "unLRG_manifest.csv"
        if argv[1] == "train":
            seed = int(argv[2])
        elif argv[1] == "test":
            seed = int(argv[2])
    else:
        print("Expected 3 inputs: random seed (e.g. 8901), rotation factor (e.g. 10) and the architectures file.")
        sys.exit(0)
    
    print("Splitting train/validation/test data...")
    data_name = data_manifest.split(".")
    # Creates 2 dictionaries: one for train/val/test and one for class labels
    if os.path.isfile("partition_"+str(seed)+"_"+str(rotate_factor)+"_"+str(train_size)+"_"+str(val_size)+"_"+data_name[0]+".json") == False:
        partition, labels = make_splits(rotate_factor, data_path, data_manifest, seed, train_size=train_size, val_size=val_size, img_rows = img_rows, img_cols = img_cols)
        json.dump(partition, open("partition_"+str(seed)+"_"+str(rotate_factor)+"_"+str(train_size)+"_"+str(val_size)+"_"+data_name[0]+".json", "w"))
        json.dump(labels, open("labels_"+str(seed)+"_"+str(rotate_factor)+"_"+str(train_size)+"_"+str(val_size)+"_"+data_name[0]+".json", "w"))
    else:
        partition = json.load(open("partition_"+str(seed)+"_"+str(rotate_factor)+"_"+str(train_size)+"_"+str(val_size)+"_"+data_name[0]+".json"))
        labels = json.load(open("labels_"+str(seed)+"_"+str(rotate_factor)+"_"+str(train_size)+"_"+str(val_size)+"_"+data_name[0]+".json"))
    save_test_labels(seed, results_path, data_path, model_path, partition, labels, img_rows, img_cols, train_size, val_size)

if __name__ == "__main__":
    main(sys.argv)

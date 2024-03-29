# '''
# This script will train and test all the models specified in the architectures file on the data specified in data_path and in the data manifest.
# The train/validation/test splits is made based on the train_size and val_size arguments. Sources are randomly selected, depending on the random seed given.
# If you'd like to train on a specific subset, create a manifest with only those items in and

# -h --help           --  Displays this docstring.
# -a --architectures  --  Path to and filename of the list of architectures you wish to train. Make sure the architectures listed have been described in the model_architectures.py file.
# -c --num_classes    --  Number of classes. Default is 4.
# -d --data_path      --  Path to the all the data, split into subdirectories per class.
# -f --data_manifest  --  Data manifest that contains the file name and class label of all the images that will be used.
# -i --img_size       --  Image size as a single int or two ints seperated by a comma for rows and columns. Default is 300,300.
# -m --model_path     --  Path to where the resulting models will be stored or new models will be loaded.
# -o --only_test      --  Boolean value to set whether to only test the model given. Default value is False.
# -p --results_path   --  Path to where the results will be stored.
# -r --rotate         --  The increments in degrees by which each individual image will be rotated. Default value is 15 degrees, leading to 360/15 = 24 augmentations per image.
# -s --seed           --  The random seed number for to ensure random processes are reproducible given the same seed. Used for weight init as well.
# -t --train_size     --  Size of the training set in number of original images. Integer, default value is 250.
# -v --val_size       --  Size of the validation set in number of original images. Integer, default value is 100.
# -X --mnist          --  Trains the models on MNIST data
# '''
import sys, getopt
from downloads import download_unLRG, download_LRG
from read_architectures import read_architectures
from results import generate_figures
from make_splits import make_splits
from train_test import train, test, validation, train_mnist, test_mnist, time_test, model_parameters, get_flops
import numpy as np
import json
import os.path
import os
import argparse

def main(argv):
    parser = argparse.ArgumentParser(description="Train and compare different architectures to each other on radio galaxy images.", 
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--seed", type=int, default="1826", help="Random seed to initialize")
    parser.add_argument("--model_path", type=str, default="models_corrections/", help="Path to models")
    parser.add_argument("--data_path", type=str, default="FITS_300/", help="Path to images")
    parser.add_argument("--final", type=bool, default=True, help="Final model or best validation")
    parser.add_argument("--results_path", type=str, default="rg_results_corrections/", help="Path to results")
    parser.add_argument("--bin_thresh", type=bool, default=False, help="Binary threshold of images") 
    parser.add_argument("--test", type=str, default="False", help="Only test the models")
    parser.add_argument("--rotate", type=int, default=15, help="Angle of rotational augmentation") 
    parser.add_argument("--train_size", type=int, default=250, help="Training set size") 
    parser.add_argument("--val_size", type=int, default=100, help="Validation set size") 
    parser.add_argument("--img_rows", type=int, default=300, help="Image row size") 
    parser.add_argument("--img_cols", type=int, default=300, help="Image column size")
    parser.add_argument("--data", type=str, default="unLRG_manifest", help="Dataset to run on")
    parser.add_argument("--data_manifest", type=str, default="unset", help="Data manifest with all images and their classes")
    parser.add_argument("--architectures_file", type=str, default="architectures.txt", help="File with list of architectures to train or test")    
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes to be classified")
    parser.add_argument("--early_stopping", type=int, default=-1, help="Adds early stopping with given epoch patience, default is off")
    try:
        opt = parser.parse_known_args()[0]
    except:
        print("")
        parser.print_help()
        sys.exit(0)
    rotate_factor = opt.rotate
    seed = opt.seed
    architectures_file = opt.architectures_file
    results_path = opt.results_path
    os.system("mkdir -p " + results_path)
    data_path = opt.data_path
    model_path = opt.model_path
    os.system("mkdir -p " + model_path)
    train_size = opt.train_size
    val_size = opt.val_size
    img_rows = opt.img_rows
    img_cols = opt.img_cols
    num_classes = opt.num_classes
    MNIST_test = False
    # Manifest is a list of files and their classes
    if opt.data == "unLRG_manifest":
        data_manifest = "unLRG_manifest.csv"
    elif opt.data == "LRG_manifest":
         data_manifest = "LRG_manifest.csv"
    elif opt.data == "MNIST":
        MNIST_test = True
        num_classes = 10
    elif opt.data_manifest != "unset":
        data_manifest = opt.data_manifest
    else:
        print("Please set the data manifest (list of files and their respective classes in CSV format)")
        sys.exit(0)
    # if opt.test != "True" or opt.test != "False":
    only_test = opt.test
    # elif opt.test == "True":
        # only_test = "True"
    # elif opt.test == "False":
        # only_test = "False"
    
    print("Splitting train/validation/test data...")
    if MNIST_test == False:
        data_name = data_manifest.split(".")
        # Creates 2 dictionaries: one for train/val/test and one for class labels
        if os.path.isfile("partition_"+str(seed)+"_"+str(rotate_factor)+"_"+str(train_size)+"_"+str(val_size)+"_"+data_name[0]+".json") == False:
            partition, labels = make_splits(rotate_factor, data_path, data_manifest, seed, train_size=train_size, val_size=val_size, img_rows = img_rows, img_cols = img_cols, bin_thresh= opt.bin_thresh)
            json.dump(partition, open("partition_"+str(seed)+"_"+str(rotate_factor)+"_"+str(train_size)+"_"+str(val_size)+"_"+data_name[0]+".json", "w"))
            json.dump(labels, open("labels_"+str(seed)+"_"+str(rotate_factor)+"_"+str(train_size)+"_"+str(val_size)+"_"+data_name[0]+".json", "w"))
        else:
            partition = json.load(open("partition_"+str(seed)+"_"+str(rotate_factor)+"_"+str(train_size)+"_"+str(val_size)+"_"+data_name[0]+".json"))
            labels = json.load(open("labels_"+str(seed)+"_"+str(rotate_factor)+"_"+str(train_size)+"_"+str(val_size)+"_"+data_name[0]+".json"))
    # Read in architecture names and hyperparams to be trained
    architectures = read_architectures(architectures_file)
    results_dict = {}
    # Train and test
    for arch in architectures.keys():
        dictionary_temp = {}        
        if MNIST_test == True:
            train_mnist(arch ,architectures[arch], seed, results_path, data_path, model_path, img_rows, img_cols)
            cm, ncm, mpca, time_dif = test_mnist(arch ,architectures[arch], seed, results_path, data_path, model_path, img_rows, img_cols)
            dictionary_temp["MPCA"] = mpca
            dictionary_temp["Norm. Confusion Matrix"] = ncm
            dictionary_temp["Confusion Matrix"] = cm
            dictionary_temp["Time"] = time_dif
            results_dict[arch] = dictionary_temp
            cm, ncm, mpca, time_dif = test_mnist(arch ,architectures[arch], seed, results_path, data_path, model_path, img_rows, img_cols,final=True)
            dictionary_temp["MPCA"] = mpca
            dictionary_temp["Norm. Confusion Matrix"] = ncm
            dictionary_temp["Confusion Matrix"] = cm
            dictionary_temp["Time"] = time_dif
            results_dict[arch+"_final"] = dictionary_temp
        elif only_test == "False":
            print("Starting Training")
            train(arch ,architectures[arch], seed, results_path, data_path, model_path, partition, labels, img_rows, img_cols, opt.early_stopping)
            cm, ncm, mpca, time_dif = test(arch ,architectures[arch], seed, results_path, data_path, model_path, partition, labels, img_rows, img_cols)
            dictionary_temp["MPCA"] = mpca
            dictionary_temp["Norm. Confusion Matrix"] = ncm
            dictionary_temp["Confusion Matrix"] = cm
            dictionary_temp["Time"] = time_dif
            results_dict[arch] = dictionary_temp
            cm, ncm, mpca, time_dif = test(arch ,architectures[arch], seed, results_path, data_path, model_path, partition, labels, img_rows, img_cols,final=True)
            dictionary_temp["MPCA"] = mpca
            dictionary_temp["Norm. Confusion Matrix"] = ncm
            dictionary_temp["Confusion Matrix"] = cm
            dictionary_temp["Time"] = time_dif
            results_dict[arch+"_final"] = dictionary_temp
        elif only_test == "True":
            print("Starting Testing")
            validation(arch ,architectures[arch], seed, results_path, data_path, model_path, partition, labels, img_rows, img_cols)
            cm, ncm, mpca, time_dif = test(arch ,architectures[arch], seed, results_path, data_path, model_path, partition, labels, img_rows, img_cols)
            dictionary_temp["MPCA"] = mpca
            dictionary_temp["Norm. Confusion Matrix"] = ncm
            dictionary_temp["Confusion Matrix"] = cm
            dictionary_temp["Time"] = time_dif
            results_dict[arch] = dictionary_temp
            cm, ncm, mpca, time_dif = test(arch ,architectures[arch], seed, results_path, data_path, model_path, partition, labels, img_rows, img_cols,final=True)
            dictionary_temp["MPCA"] = mpca
            dictionary_temp["Norm. Confusion Matrix"] = ncm
            dictionary_temp["Confusion Matrix"] = cm
            dictionary_temp["Time"] = time_dif
            results_dict[arch+"_final"] = dictionary_temp
        elif only_test == "time":
            print("Starting Timing")
            time_arr, ips_arr, time_dif, time_std, ips, ips_std = time_test(arch ,architectures[arch], seed, results_path, data_path, model_path, partition, labels, img_rows, img_cols)
            dictionary_temp["IPS"] = ips
            dictionary_temp["Time"] = time_dif
            dictionary_temp["T_arr"] = time_arr
            dictionary_temp["IPS_arr"] = ips_arr
            dictionary_temp["Time_std"] = time_std
            dictionary_temp["IPS_std"] = ips_std
            results_dict[arch] = dictionary_temp
        elif only_test == "parameters":
            dense, conv, flatten, trainable_params = model_parameters(arch ,architectures[arch], seed, model_path, img_rows, img_cols)
            dictionary_temp["conv"] = conv
            dictionary_temp["dense"] = dense
            dictionary_temp["flatten"] = flatten
            dictionary_temp["trainable_params"] = trainable_params
            results_dict[arch] = dictionary_temp
        elif only_test == "flops":
            flops = get_flops(arch ,architectures[arch], seed, model_path, img_rows, img_cols)
            dictionary_temp["flops"] = flops            
            results_dict[arch] = dictionary_temp
    print(results_dict)
    if opt.test == "time":
        f = open(results_path+"/times_new.txt","a")
        f.write(str(results_dict)+'\n')
        f.close()
    elif opt.test == "parameters":
        f = open(results_path+"/parameter_summary.txt","a")
        f.write(str(results_dict)+'\n')
        f.close()
    elif opt.test == "flops":
        f = open(results_path+"/flops_summary.txt","a")
        f.write(str(results_dict)+'\n')
        f.close()
    else:
        f = open(results_path+"/results_dict_corrections_overfit.txt","a")
        f.write(str(results_dict)+'\n')
        f.close()
        
    # Make plots
    # generate_figures(results_dict, results_path, partition, labels)

if __name__ == "__main__":
    main(sys.argv)

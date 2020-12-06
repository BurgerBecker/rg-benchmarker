'''
This script will train and test all the models specified in the architectures file on the data specified in data_path and in the data manifest.
The train/validation/test splits is made based on the train_size and val_size arguments. Sources are randomly selected, depending on the random seed given.
If you'd like to train on a specific subset, create a manifest with only those items in and

-h --help           --  Displays this docstring.
-a --architectures  --  Path to and filename of the list of architectures you wish to train. Make sure the architectures listed have been described in the model_architectures.py file.
-c --num_classes    --  Number of classes. Default is 4.
-d --data_path      --  Path to the all the data, split into subdirectories per class.
-f --data_manifest  --  Data manifest that contains the file name and class label of all the images that will be used.
-i --img_size       --  Image size as a single int or two ints seperated by a comma for rows and columns. Default is 300,300.
-m --model_path     --  Path to where the resulting models will be stored or new models will be loaded.
-o --only_test      --  Boolean value to set whether to only test the model given. Default value is False.
-p --results_path   --  Path to where the results will be stored.
-r --rotate         --  The increments in degrees by which each individual image will be rotated. Default value is 15 degrees, leading to 360/15 = 24 augmentations per image.
-s --seed           --  The random seed number for to ensure random processes are reproducible given the same seed. Used for weight init as well.
-t --train_size     --  Size of the training set in number of original images. Integer, default value is 250.
-v --val_size       --  Size of the validation set in number of original images. Integer, default value is 100.
-X --mnist          --  Trains the models on MNIST data
'''
import sys, getopt
from downloads import download_unLRG, download_LRG
from read_architectures import read_architectures
from results import generate_figures
from make_splits import make_splits
# from train_test import train, test, train_mnist, test_mnist, time_test
import numpy as np
import json
import os.path

def main(argv):
    MNIST_test = False
    only_test = "False"
    bin_thresh = False
    short_options = "hs:r:a:p:d:m:t:v:i:o:f:x:c:"
    long_options = ["help", "seed=", "rotate=", "architectures=", "results_path=", "data_path=", "model_path=", "train_size=", "val_size=", "img_size=","only_test=","data_manifest=","mnist=","number_classes="]
    argument_list = argv[1:]
    try:
        arguments, values = getopt.getopt(argument_list, short_options, long_options)
    except getopt.error as err:
        # Output error, and return with an error code
        print (str(err))
        sys.exit(2)
    print(arguments)
    rotate_factor = 15
    seed = 8901
    architectures_file = "architectures.txt"
    results_path = "rg_class_experiment_results/"
    data_path = "FITS/"
    model_path = "models/"
    train_size = 250
    val_size = 100
    img_rows = 300
    img_cols = 300
    num_classes = 4
    # Manifest is a list of files and their classes
    data_manifest = "unLRG_manifest.csv"
    only_test = False
    MNIST_test = False
    for current_argument, current_value in arguments:
        if current_argument in ("-a","--architectures"):
            architectures_file = current_value
        elif current_argument in ("-c","--num_classes"):
            num_classes = int(current_value)
        elif current_argument in ("-d","--data_path"):
            data_path = current_value
        elif current_argument in ("-f", "--data_manifest"):
            data_manifest = current_value
        elif current_argument in ("-h", "--help"):
            print(__doc__)
        elif current_argument in ("-i","--img_size"):
            current_value = current_value.rstrip("()[]")
            if ',' in current_value:
                x = current_value.split(',')
                img_rows = int(x[0])
                img_cols = int(x[1])
            else:
                img_rows = int(current_value)
        elif current_argument in ("-m", "--model_path"):
            model_path = current_value
        elif current_argument in ("-o", "--only_test"):
            if current_value == "False":
                only_test = False
            else:
                only_test = True
        elif current_argument in ("-p", "--results_path"):
            results_path = current_value
        elif current_argument in ("-r","--rotate"):
            rotate_factor = int(current_value)
        elif current_argument in ("-s","--seed"):
            seed = int(current_value)
        elif current_argument in ("-t","--train_size"):
            train_size = int(current_value)
        elif current_argument in ("-v", "--val_size"):
            val_size = int(current_value)
        elif current_argument in ("-X", "--mnist"):
            if current_value == "False":
                MNIST_test = False
            else:
                MNIST_test = True
    # if len(argv) == 1:
    #     print("No arguments given, default random seed (8901), rotation factor (15 deg) and architectures file will be used.")
    #     rotate_factor = 15
    #     seed = 1826#8901
    #     architectures_file = "architectures.txt"
    #     results_path = "rg_class_experiment_results/"
    #     data_path = "FITS_300/"
    #     model_path = "models/"
    #     train_size = 250
    #     val_size = 100
    #     img_rows = 300
    #     img_cols = 300
    #     # Manifest is a list of files and their classes
    #     data_manifest = "unLRG_manifest.csv"
    # elif len(argv) == 12:
    #     seed = int(argv[1])
    #     rotate_factor = int(argv[2])
    #     architectures_file = argv[3]
    #     results_path = argv[4]
    #     data_path = argv[5]
    #     model_path = argv[6]
    #     data_manifest = argv[7]
    #     train_size = int(argv[8])
    #     val_size = int(argv[9])
    #     img_rows = int(argv[10])
    #     img_cols = img_rows
    #     bin_thresh = argv[11]
    # elif len(argv) == 2:
    #     if argv[1] == "mnist":
    #         MNIST_test = True
    #     elif argv[1] == "test":
    #         only_test = "default"
    #         rotate_factor = 15
    #         seed = 8901
    #         architectures_file = "architectures.txt"
    #         results_path = "rg_class_experiment_results/"
    #         data_path = "FITS_300/"
    #         model_path = "models/"
    #         train_size = 250
    #         val_size = 100
    #         img_rows = 300
    #         img_cols = 300
    #         # Manifest is a list of files and their classes
    #         data_manifest = "unLRG_manifest.csv"
    #     elif argv[1] == "time":
    #         only_test = "time"
    #         rotate_factor = 15
    #         seed = 8901
    #         architectures_file = "architectures.txt"
    #         results_path = "rg_class_experiment_results/"
    #         data_path = "FITS_300/"
    #         model_path = "models/"
    #         train_size = 250
    #         val_size = 100
    #         img_rows = 300
    #         img_cols = 300
    #         # Manifest is a list of files and their classes
    #         data_manifest = "unLRG_manifest.csv"
    # elif len(argv) == 3 or len(argv) == 4:
    #     rotate_factor = 15
    #     architectures_file = "architectures.txt"
    #     results_path = "result_unLRG/"
    #     data_path = "FITS_300/"
    #     model_path = "models/"
    #     train_size = 250
    #     val_size = 100
    #     data_manifest = "unLRG_manifest.csv"
    #     img_rows = 300
    #     img_cols = 300
    #     if argv[1] == "train":
    #         seed = int(argv[2])
    #         if len(argv) == 4:
    #             bin_thresh = argv[3]
    #     elif argv[1] == "test":
    #         only_test = "default"
    #         seed = int(argv[2])
    # elif len(argv) == 7:
    #     rotate_factor = 15
    #     architectures_file = "architectures.txt"
    #     results_path = "result_unLRG/"
    #     data_path = "FITS_300/"
    #     model_path = "models/"
    #     train_size = 250
    #     val_size = 100
    #     data_manifest = "unLRG_manifest.csv"
    #     img_rows = 300
    #     img_cols = 300
    #     if argv[1] == "train":
    #         seed = int(argv[2])
    #     elif argv[1] == "test":
    #         only_test = "default"
    #         seed = int(argv[2])
    #         model_path = argv[3]
    #         results_path = argv[4]
    #         train_size = int(argv[5])
    #         val_size = int(argv[6])
    # else:
    #     print("Expected 3 inputs: random seed (e.g. 8901), rotation factor (e.g. 10) and the architectures file.")
    #     sys.exit(0)
    
    print("Splitting train/validation/test data...")
    data_name = data_manifest.split(".")
    # Creates 2 dictionaries: one for train/val/test and one for class labels
    if os.path.isfile("partition_"+str(seed)+"_"+str(rotate_factor)+"_"+str(train_size)+"_"+str(val_size)+"_"+data_name[0]+".json") == False:
        partition, labels = make_splits(rotate_factor, data_path, data_manifest, seed, train_size=train_size, val_size=val_size, img_rows = img_rows, img_cols = img_cols, bin_thresh= bin_thresh)
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
            train_mnist(arch ,architectures[arch], seed, results_path, data_path, model_path, partition, labels, img_rows, img_cols)
            cm, ncm, mpca, time_dif = test_mnist(arch ,architectures[arch], seed, results_path, data_path, model_path, partition, labels, img_rows, img_cols)
            dictionary_temp["MPCA"] = mpca
            dictionary_temp["Norm. Confusion Matrix"] = ncm
            dictionary_temp["Confusion Matrix"] = cm
            dictionary_temp["Time"] = time_dif
            results_dict[arch] = dictionary_temp
            cm, ncm, mpca, time_dif = test_mnist(arch ,architectures[arch], seed, results_path, data_path, model_path, partition, labels, img_rows, img_cols,final=True)
            dictionary_temp["MPCA"] = mpca
            dictionary_temp["Norm. Confusion Matrix"] = ncm
            dictionary_temp["Confusion Matrix"] = cm
            dictionary_temp["Time"] = time_dif
            results_dict[arch+"_final"] = dictionary_temp
        elif only_test == False:
            train(arch ,architectures[arch], seed, results_path, data_path, model_path, partition, labels, img_rows, img_cols)
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
        elif only_test == True:
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
            time_arr, ips_arr, time_dif, time_std, ips, ips_std = time_test(arch ,architectures[arch], seed, results_path, data_path, model_path, partition, labels, img_rows, img_cols)
            dictionary_temp["IPS"] = ips
            dictionary_temp["Time"] = time_dif
            dictionary_temp["T_arr"] = time_arr
            dictionary_temp["IPS_arr"] = ips_arr
            dictionary_temp["Time_std"] = time_std
            dictionary_temp["IPS_std"] = ips_std
            results_dict[arch] = dictionary_temp
    print(results_dict)
    # Make plots
    # generate_figures(results_dict, results_path, partition, labels)

if __name__ == "__main__":
    main(sys.argv)

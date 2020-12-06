import sys
from downloads import download_unLRG, download_LRG
from read_architectures import read_architectures
from results import generate_figures
import make_splits

def main(argv):
    if len(argv) == 1:
        print("No arguments given, default random seed (8901), rotation factor (10 deg) and architectures file will be used.")
        rotate_factor = 10
        seed = 8901
        architectures_file = "architectures.txt"
        results_path = "rg_class_experiment_results/"
        data_path = "FITS/"
    elif len(argv) == 3:
        seed = int(argv[1])
        rotate_factor = int(argv[2])
        architectures_file = argv[3]
        results_path = argv[4]
        data_path = argv[5]
    else:
        print("Expected 3 inputs: random seed (e.g. 8901), rotation factor (e.g. 10) and the architectures file.")
        sys.exit(0)
    
    print("Starting LRG download...")
    errors = download_LRG(data_path)
    print("LRG download complete, with "+str(errors)+" download errors.\n")
    print("Starting unLRG download...")
    errors = download_unLRG(data_path)
    print("unLRG download complete, with "+str(errors)+" download errors.\n")

    print("Splitting train/validation/test data...")
    # Creates 2 dictionaries: one for train/val/test and one for class labels
    partition, labels = make_splits(rotate_factor, data_path, seed)
    # Read in architecture names and hyperparams to be trained
    architectures = read_architectures(architectures_file)
    # Train and test
    for arch in architectures.keys():
        print("Training "+arch)
        train(architectures[arch], seed, results_path, partition, labels)
        # Can't testing be part of this? ^
        print("Testing "+arch)
        test(architectures[arch], seed, results_path, partition, labels)
    # Make plots
    generate_figures(architectures, results_path, partition, labels)

if __name__ == "__main__":
    main(sys.argv)
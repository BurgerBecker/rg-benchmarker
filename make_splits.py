import numpy as np
import csv
from read_architectures import get_class_label
import random
from astropy.io import fits
import os.path
from skimage.transform import rotate
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg

def convert_rgb_to_grey(source):
    if len(source.shape) == 3:
        source_grey = 0.3*source[:,:,0] + 0.59*source[:,:,1] + 0.11*source[:,:,2]
    else:
        print("More than 3 channels in image. Custom greyscale conversion advised.")
        source_grey = np.average(source, axis=2)
    return source_grey

def preproc_fast(image, sigma, bin_thresh=False):    
    mean = np.average(image)
    threshold = np.std(image)*sigma
    above = np.where(image > mean+threshold,image,0.0)
    if bin_thresh:
        max_val = np.max(image)
        above = np.where(image > 0.8*max_val,1.0,image)
    return above

def rotate_source(split, partition, labels_aug, label, source, rotate_factor, data_path, label_map, extension=".fits",sub_class_map={},num_classes=4, img_rows=300, img_cols=300, bin_thresh=False):
    steps = [i for i in range(0,360,int(rotate_factor))]
    skip = False
    # Read source
    if os.path.isfile(data_path+"/"+label_map+"/"+source+".fits") == True:
        image_file = fits.open(data_path+"/"+label_map+"/"+source+".fits")
        image = image_file[0].data
    elif os.path.isfile(data_path+"/"+label_map+"/"+source+extension) == True:
        image = plt.imread(data_path+"/"+label_map+"/"+source+extension)
    else:
        print("File not found: "+data_path+"/"+label_map+"/"+source+extension)
        print("Skipping...")
        skip = True
        return partition, labels_aug, skip
    # Checking for NaN's
    if np.isnan(image).all():
        print("File:"+source+" only consists of NaN values.\n Skipping...")
        skip = True
        return partition, labels_aug, skip
    # Removing NaN values           
    image[np.isnan(image)] = 0
    # Convert RGB to single channel
    if len(image.shape) >= 3:
        image = convert_rgb_to_grey(image)
    # Normalize data
    image = (image - np.min(image))/(np.max(image)-np.min(image))
    img_rowsf, img_colsf = image.shape
    if img_rows != img_rowsf or img_cols != img_colsf:
        skip = True
        print("File:"+source+" has shape: ",image.shape)
        print("expected size: "+ str(img_rows)+", "+str(img_cols))
        return partition, labels_aug, skip 
    # Preprocess source
    above = preproc_fast(image, 3, bin_thresh=bin_thresh)
    # Step through number of rotations and save each rotation
    for step in steps:
        if os.path.isfile(data_path+"/"+label_map+"/"+source+"_rot_"+str(step)+".fits") == False:
            rot = rotate(above.copy(),step)
            # Save rotated source
            outfile = source+"_rot_"+str(step)+".fits"
            hdu = fits.PrimaryHDU(rot)
            hdu.writeto(data_path+"/"+label_map+"/"+outfile,overwrite=True)
        # if os.path.isfile(data_path+label_map+"/"+source+"_rot_"+str(step)+".png") == False:
        #     rot = rotate(above.copy(),step)
        #     # Save rotated source
        #     # plt.figure(1)
        #     # plt.imshow(rot,cmap="gray");
        #     # plt.savefig(data_path+label_map+"/"+source+"_rot_"+str(step)+".png")
        #     mpimg.imsave(data_path+label_map+"/"+source+"_rot_"+str(step)+".png",rot)
        partition[split].append(source+"_rot_"+str(step))
        if label < num_classes:
            labels_aug[source+"_rot_"+str(step)] = label
        else:
            labels_aug[source+"_rot_"+str(step)] = sub_class_map[label]
    return partition, labels_aug, skip

def make_splits(rotate_factor, data_path, data_manifest, seed, train_size=250,val_size=250, img_rows=300,img_cols=300,bin_thresh=False):
    """Read sources from data manifest stored at the data path, augment data and split into training, validation and test sets.
    rotate_factor   --  gives the steps in degrees between rotations. 10 degree rotate_factor will create 36 rotations at 10 deg intervals
    data_path       --  path to where the data is stored
    data_manifest   --  CSV file with a list of all the file names and their class labels as a number
    seed            --  random seed for the experiment, different seed will return different partition
    train_size      --  size of the training set
    val_size        --  size of the validation set
    img_rows        --  number of rows in the image
    img_cols        --  number of columns in the image
    bin_thresh      --  whether or not to binary threshold the data
    """
    partition = {"train":[],"validation":[],"test":[]}
    labels = {}
    f = open(data_manifest,'r')
    line_count = 0    
    label_map = get_class_label()
    num_classes = len(label_map.keys())    
    # If there is a sub-class, we need to have equal representation in the training set
    sub_class = 0
    fr0 = 0
    sub_class_labels = []
    sub_class_map = {}
    for line in f:
        # Assuming there is a column title
        if line_count != 0:     
            line = line.rstrip()
            x = line.split(',')
            # labels[x[0]] = int(x[1][0])-1
            # Will need to update this to add full subclass support
            if x[1] != '1F':
                labels[x[0]] = int(x[1][0])-1
            # else:
            #     labels[x[0]] = num_classes
            #     sub_class = 1
            #     sub_class_labels = [int(x[1][0])-1,num_classes]
            #     sub_class_map = {num_classes:int(x[1][0])-1}
            #     label_map[num_classes+1] = label_map[int(x[1][0])]
            #     fr0+=1
        line_count+=1
    f.close()
    # print(fr0)
    keys = list(labels.keys())
    random.seed(seed)
    random.shuffle(keys)
    train = np.zeros([num_classes+sub_class])
    validation = np.zeros([num_classes+sub_class])
    labels_aug = {}
    counter = 0
    for key in keys:
        if counter%np.floor(len(keys)/20) == 0:
            print(str(np.round(counter/len(keys)*100,0))+"%")
        counter+=1
        if train[labels[key]] < train_size:
            partition, labels_aug, skip = rotate_source("train", partition, labels_aug, labels[key], key, rotate_factor, data_path, label_map[labels[key]+1], sub_class_map=sub_class_map, num_classes=num_classes,img_rows=img_rows,img_cols=img_cols,bin_thresh=bin_thresh)
            if skip == False:
                train[labels[key]]+=1
            else:
                continue
        # elif train[labels[key]] < int(train_size/2) and labels[key] in sub_class_labels:
        #     partition, labels_aug, img_rows, img_cols, skip = rotate_source("train", partition, labels_aug, labels[key], key, rotate_factor, data_path, label_map[labels[key]+1], sub_class_map=sub_class_map, num_classes=num_classes)
        #     if skip == False:
        #         train[labels[key]]+=1
        #     else:
        #         continue
        elif validation[labels[key]] < val_size:
            partition, labels_aug, skip = rotate_source("validation", partition, labels_aug, labels[key], key, rotate_factor, data_path, label_map[labels[key]+1], sub_class_map=sub_class_map, num_classes=num_classes,img_rows=img_rows,img_cols=img_cols,bin_thresh=bin_thresh)
            if skip == False:
                validation[labels[key]]+=1
            else:
                continue
        # elif validation[labels[key]] < int(val_size/2) and labels[key] in sub_class_labels:
        #     partition, labels_aug, _, _, skip = rotate_source("validation", partition, labels_aug, labels[key], key, rotate_factor, data_path, label_map[labels[key]+1], sub_class_map=sub_class_map, num_classes=num_classes)
        #     if skip == False:
        #         validation[labels[key]]+=1
        #     else:
        #         continue
        else:
            if os.path.isfile(data_path+"/"+label_map[labels[key]+1]+"/"+key+".fits") != False:
                image_file = fits.open(data_path+"/"+label_map[labels[key]+1]+"/"+key+".fits")
                image = image_file[0].data
                if np.isnan(image).all():
                    continue
                image[np.isnan(image)] = 0
                image = (image - np.min(image))/(np.max(image)-np.min(image))
                above = preproc_fast(image, 3, bin_thresh=bin_thresh)
                outfile = key+"_clipped"+".fits"
                hdu = fits.PrimaryHDU(above)
                hdu.writeto(data_path+"/"+label_map[labels[key]+1]+"/"+outfile,overwrite=True)
            else:
                continue
            partition["test"].append(key+"_clipped")
            labels_aug[key+"_clipped"] = labels[key]
    print(train)
    print(validation)
    return partition, labels_aug
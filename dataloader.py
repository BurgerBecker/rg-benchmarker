import numpy as np
import tensorflow.keras as keras
from astropy.io import fits
from skimage.transform import rotate
import sys
import matplotlib.pyplot as plt

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras, adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly'
    def __init__(self, list_IDs, labels, data_path, label_map, seed, input_shape, img_rows=300, img_cols=300, batch_size=32, dim=(300,300), n_channels=1,
                 n_classes=4, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.seed = seed
        np.random.seed(self.seed)
        self.list_IDs = list_IDs
        self.labels = labels
        self.data_path = data_path
        self.label_map = label_map
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.input_shape = input_shape
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if (index+1)*self.batch_size < len(self.list_IDs):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            bs = self.batch_size
        else:
            bs = len(self.list_IDs) - index*self.batch_size
            indexes = self.indexes[index*self.batch_size:index*self.batch_size+bs]
        # # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp, bs)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, bs):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((bs, self.img_rows, self.img_cols, self.n_channels))
        y = np.zeros((bs,self.n_classes), dtype=int)
        # classes = ["COMP","FRI","FRII","BENT"]
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # if os.path.isfile(data_path+label_map+"/"+source+".fits") == True:
            image_file = fits.open(self.data_path+"/"+self.label_map[self.labels[ID]+1]+"/"+ID+".fits")
            image = image_file[0].data
            # image_rgb = plt.imread(self.data_path+self.label_map[self.labels[ID]+1]+"/"+ID+".png")
            # image = image_rgb[:,:,0]
            # if self.input_shape == 'channels_first':
                # image = image.reshape(1, self.n_channels, self.img_rows, self.img_cols)
            # else:
            image = image.reshape(1, self.img_rows, self.img_cols, self.n_channels)
            X[i,:,:,:] = image
            y[i,self.labels[ID]] = 1
        # print(y)
        return X, y
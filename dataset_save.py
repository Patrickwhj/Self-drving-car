import numpy as np
import sys
import os
from time import time
import math


def dataset_save(X,y,directory):
    CHUNK_SIZE=128
    X_len=X.shape[0]
    turns=int(math.ceil(X_len/CHUNK_SIZE))
    print("number of files: {}".format(X_len))
    print("turns: {}".format(turns))

    for turn in range(0, turns):
        
        img_data = np.zeros([1, 120, 160, 3])
        img_labels = np.zeros((1, 3), 'float')           
        
        
        img_data_temp = X[(turn*CHUNK_SIZE):((turn+1)*(CHUNK_SIZE)),:]
        img_labels_temp = y[(turn*CHUNK_SIZE):((turn+1)*(CHUNK_SIZE)),:]
        
        img_data = np.vstack((img_data, img_data_temp))
        img_labels = np.vstack((img_labels, img_labels_temp))
               
        img_data = img_data[1:, :]
        img_labels = img_labels[1:, :]
        
        print("Image array shape:" + str(img_data.shape))
        print("Label array shape: "+ str(img_labels.shape))
        file_name = str(time())

        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            np.savez(directory + '/' + file_name + '.npz', img_data=img_data, img_labels=img_labels)
        except IOError as e:
            print(e)
    
import glob
import os
import sys
import numpy as np
import img_global

def data_load(directory):
    image_array=np.zeros([1,HEIGHT,WIDTH,CHANNELS])
    label_array=np.zeros((1,OUTPUT_SHAPE),'float')
    image_data=glob.glob(directory + '/*.npz')
    
    # return the list of npz
    print("NPZ list ready")
    print("%d turns in total "%len(image_data))

    if not image_data:
        print("No training data in directory, exit")
        sys.exit()
    i = 0
    for single_npz in image_data:
        with np.load(single_npz) as data:
            print(data.keys())
            i=i+1
            print("Current npz",i)
            img_temp = data['img_data']
            img_labels_temp=data['img_labels']
        image_array = np.vstack((image_array, img_temp))
        label_array = np.vstack((label_array, img_labels_temp))
        print("The %d turn is finished" %i)
    print("Loop finished")
    X = image_array[1:, :]
    y = label_array[1:, :]
    print('Image array shape: ' + str(X.shape))
    print('Label array shape: ' + str(y.shape))
    print(np.mean(X))
    print(np.var(X))

    return X, y
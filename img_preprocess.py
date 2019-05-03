import sys
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import img_global
import dataset_load
import dataset_save

def img_aug(directory):
    X_train, y_train, X_test, y_test = img_split(directory)
    print(len(X_train))
    train_temp = X_train
    train_labels_temp = y_train
    train_temp_flipped = np.array([np.fliplr(i) for i in train_temp])
    train_labels_flipped = train_labels_temp
    for i in train_labels_flipped:
        if not all(np.array(i) - np.array([0., 1., 0.])):
            i = [0., 0., 1.]
        if not all(np.array(i) - np.array([0., 0., 1.])):
            i = [0., 1., 0.]
        else:
            pass
    X_train = np.vstack((X_train, train_temp_flipped))
    y_train = np.vstack((y_train, train_labels_flipped))

    print('X_train shape:' + str(X_train.shape))
    print('y_train shape:' + str(y_train.shape))

    images = np.empty([X_train.shape[0], 120, 160, 3])
    steers = np.empty([X_train.shape[0], 3])
    i = 0
    for index in np.random.permutation(X_train.shape[0]):
        images[i] = X_train[index]
        steers[i] = y_train[index]
        i += 1
    print('img shape:' + str(images.shape))
    print('steer shape:' + str(steers.shape))
    dataset_save.dataset_save(images, steers, "./aug_train_set")
    dataset_save.dataset_save(X_test, y_test, "./test_set")


def img_split(directory):
    # load
    image_array = np.zeros([1, HEIGHT, WIDTH, CHANNELS])
    label_array = np.zeros((1, OUTPUT), 'float')
    X,y=dataset_load(directory)

    # Split the data into a training (90) set, testing set(10) (Validation set will be split during model training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    return X_train, y_train,X_test,y_test

if __name__ == '__main__':
    img_aug('./img_data_npz')

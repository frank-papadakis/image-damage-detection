from functions import damage_creation
from functions import damage_detection
from functions import finite_volumes_split
from functions import segmentation

import shutil
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
import copy
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

### TENSORFLOW LIBRARIES ###
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
import tensorflow as tf
from keras.datasets import mnist
from tensorflow.keras import metrics
from keras.callbacks import ModelCheckpoint
from keras import backend as K
### SEEDS ###
np.random.seed(7)
tf.random.set_seed(7)
### LIBRARIES

rng=default_rng()


def main():
    num=100
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    del train_labels
    del test_labels
    train_images = train_images.reshape((-1, 28, 28))
    train_images = (train_images.astype('float32') / 255)*2-1 ### this has to be done before  -or passed as a parameter
    X_new, y_new = damage_creation.create_damage(train_images,'medium')
    
    
    X_new=X_new.reshape(-1,28,28,1)
    y_new=y_new.reshape(-1,784)
    x_train,  x_val, y_train,y_val =train_test_split(X_new,y_new,train_size=0.8)
    #history, model = damage_detection.train_model(x_train, y_train, x_val, y_val,'data\detection_model.h5py')
    model=damage_detection.load_model()
    print('loaded model')
    pred=segmentation.image_reconstruction(X_new[5].reshape(1,28,28,1),model)
    plt.imshow(pred.reshape(28,28))
    plt.show()
    
    print('done')

if __name__=='__main__':
    main()
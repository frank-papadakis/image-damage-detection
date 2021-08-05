import skimage.filters as filters
from functions import finite_volumes_split 
import numpy as np
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__




def image_reconstruction(images,model):
    shape=np.array(np.shape(images))
    test_pred=model.predict(images.reshape(-1,shape[1],shape[2],1))
    blockPrint()
    repair=np.zeros(shape=(shape[0],shape[1]*shape[2]))
    for i in range(shape[0]):
        t = filters.threshold_otsu(test_pred[i])
        newpred=test_pred[i].reshape(784)
        damaged_region=newpred>t
        damaged_region=np.where(damaged_region)[0]
        repair[i]=finite_volumes_split.temporal_loop_split(images[i].reshape(shape[1]*shape[2]), damaged_region)
    enablePrint()
    return repair
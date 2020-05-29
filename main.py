import sys
import time

from config import latent_dim, modelsPath, imageSize
from keras.optimizers import RMSprop
from model import getModels,showModels
# from visuals import visualizeDataset, visualizeReconstructedImages, computeTSNEProjectionOfLatentSpace, computeTSNEProjectionOfPixelSpace, visualizeInterpolation, visualizeArithmetics
# from datasetTools import loadDataset
import numpy as np
import tensorflow as tf
from random import randint

def trainModel(startEpoch=0):
    # create models
    print("Creating auto-encoder...")
    autoencoder = getModels()
    showModels(autoencoder,'auto_encoder')



if __name__ == '__main__':
    trainModel()


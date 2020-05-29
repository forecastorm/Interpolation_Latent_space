import os
import sys
import h5py
import cv2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.stats import norm
from sklearn import manifold
from keras.utils.vis_utils import plot_model

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Convolution2D, UpSampling2D, MaxPooling2D
from keras.models import Model
from keras.layers.advanced_activations import ELU
from keras import backend as K
from keras import objectives
from config import latent_dim, imageSize


def getModels():
    input_img = Input(shape=(imageSize, imageSize, 1))
    x = Convolution2D(32, 3, 3, border_mode='same')(input_img)
    x = ELU()(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = ELU()(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)

    # Latent space // bottleneck layer
    x = Flatten()(x)
    x = Dense(latent_dim)(x)
    z = ELU()(x)

    ##### MODEL 1: ENCODER #####
    encoder = Model(input_img, z)
    return encoder


def showModels(model,name):
    model.summary()
    name += "_plot.png"
    plot_model(model, to_file=name, show_shapes=True, show_layer_names=True)

import os
import pathlib
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
np.random.seed(123)
import itertools
import warnings
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,Input,Average,Conv2DTranspose,SeparableConv2D,dot,UpSampling2D,Add, Flatten,Concatenate,Multiply,Conv2D, MaxPooling2D,Activation,AveragePooling2D, ZeroPadding2D,GlobalAveragePooling2D,multiply,DepthwiseConv2D,ZeroPadding2D,GlobalAveragePooling2D
from keras import backend as K
from keras.layers import concatenate ,Lambda
import itertools
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy
import numpy as np
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from math import sqrt, ceil
from tqdm import tqdm_notebook as tqdm
import cv2
from sklearn.utils import shuffle
from tqdm import tqdm
import tifffile as tif
from msrf_model import msrf
from msrf_model import *
from tensorflow.keras.callbacks import *
import skimage.io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray
from msrf_loss import *
from msrf_utils import *
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session()
from glob import glob

img_dims = 1
image_size = 512
np.random.seed(42)
create_dir("files")
print(tf.__version__)

root_path = "/root/research/deep-segmentation/"
train_path = root_path + "data/"
valid_path = root_path + "data/test"

## Training
train_x = sorted(glob(os.path.join(train_path, "inputs", "*.png")))
train_y = sorted(glob(os.path.join(train_path, "labels", "*.png")))

## Shuffling
train_x, train_y = shuffling(train_x, train_y)
train_x = train_x
train_y = train_y

print(train_path)
print(valid_path)

## Validation
valid_x = sorted(glob(os.path.join(valid_path, "inputs", "*.png")))
valid_y = sorted(glob(os.path.join(valid_path, "labels", "*.png")))


print("final training set length",len(train_x),len(train_y))
print("final valid set length",len(valid_x),len(valid_y))

import random


X_tot_val = [get_image(sample_file,image_size,image_size) for sample_file in valid_x]
X_val,edge_x_val = [],[]
print('X total length', len(X_tot_val))
for i in range(0,len(valid_x)):
    X_val.append(X_tot_val[i][0])
    edge_x_val.append(X_tot_val[i][1])
X_val = np.array(X_val).astype(np.float32)
edge_x_val = np.array(edge_x_val).astype(np.float32)
print('edge_x_val ', edge_x_val.shape)
edge_x_val  =  np.expand_dims(edge_x_val,axis=img_dims)
Y_tot_val = [get_image(sample_file,image_size,image_size,gray=True) for sample_file in valid_y]
Y_val,edge_y = [],[]
for i in range(0,len(valid_y)):
    Y_val.append(Y_tot_val[i][0])
Y_val = np.array(Y_val).astype(np.float32)
print('Y_val ', Y_val.shape)
Y_val  =  np.expand_dims(Y_val,axis=img_dims)

def train(epochs, batch_size, output_dir, model_save_dir):
    
    batch_count = int(len(train_x) / batch_size)
    max_val_dice= -1
    G = msrf()
    # G.summary()
    optimizer = get_optimizer()
    G.compile(optimizer = optimizer, loss = {'x':seg_loss,'edge_out':'binary_crossentropy','pred4':seg_loss,'pred2':seg_loss},loss_weights={'x':2.,'edge_out':1.,'pred4':1. , 'pred2':1.})
    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15,batch_size)
        #sp startpoint
        for sp in range(0,batch_count,1):
            if (sp+1)*batch_size>len(train_x):
                batch_end = len(train_x)
            else:
                batch_end = (sp+1)*batch_size
            X_batch_list = train_x[(sp*batch_size):batch_end]
            Y_batch_list = train_y[(sp*batch_size):batch_end]
            X_tot = [get_image(sample_file,image_size,image_size) for sample_file in X_batch_list]
            X_batch,edge_x = [],[]
            for i in range(0,batch_size):
                X_batch.append(X_tot[i][0])
                edge_x.append(X_tot[i][1])
            X_batch = np.array(X_batch).astype(np.float32)
            edge_x = np.array(edge_x).astype(np.float32)
            Y_tot = [get_image(sample_file,image_size,image_size, gray=True) for sample_file in Y_batch_list]
            Y_batch,edge_y = [],[]
            for i in range(0,batch_size):
                Y_batch.append(Y_tot[i][0])
                edge_y.append(Y_tot[i][1])
            Y_batch = np.array(Y_batch).astype(np.float32)
            edge_y = np.array(edge_y).astype(np.float32)
            print('Y_batch', Y_batch.shape, edge_y.shape, edge_x.shape)
            Y_batch  =  np.expand_dims(Y_batch,axis=img_dims).reshape(-1,image_size,image_size,1)
            edge_y = np.expand_dims(edge_y,axis=img_dims).reshape(-1,image_size,image_size,1)
            edge_x = np.expand_dims(edge_x,axis=img_dims).reshape(-1,image_size,image_size,1)
            # X_batch = X_batch.reshape(-1,image_size,image_size,1)
            G.train_on_batch([X_batch,edge_x],[Y_batch,edge_y,Y_batch,Y_batch])

        y_pred,_,_,_ = G.predict([X_val,edge_x_val],batch_size=5)
        y_pred = (y_pred >=0.5).astype(int)
        res = mean_dice_coef(Y_val,y_pred)
        if(res > max_val_dice):
            max_val_dice = res
            G.save('kdsb_ws.h5')
            print('New Val_Dice HighScore',res)            
            
model_save_dir = './model/'
output_dir = './output/'
train(125,4,output_dir,model_save_dir)
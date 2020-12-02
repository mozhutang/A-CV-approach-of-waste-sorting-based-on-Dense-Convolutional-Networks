
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from PIL import Image

import os
from tqdm import tqdm 

# the folder from train.zip file
train_dir = '/home/ubuntu/train/'

# a folder where split and resized data will be stored
data_dir = '/home/ubuntu/data/'


# # Load constant train-val split

T = pd.read_csv('../train_val_split/train_metadata.csv')
V = pd.read_csv('../train_val_split/val_metadata.csv')


# # Create directories for different categories

os.mkdir(data_dir + 'train_no_resizing')
for i in range(1, 256 + 1):
    os.mkdir(data_dir + 'train_no_resizing/' + str(i))


os.mkdir(data_dir + 'val_no_resizing')
for i in range(1, 256 + 1):
    os.mkdir(data_dir + 'val_no_resizing/' + str(i))


# # val. images

val_size = len(V)
val_size

# RGB images
for i, row in tqdm(V.loc[V.channels == 3].iterrows()):
    # get image
    file_path = os.path.join(train_dir, row.directory, row.img_name)
    image = Image.open(file_path)
    
    # save
    save_path = os.path.join(data_dir, 'val_no_resizing', str(row.category_number), row.img_name)
    image.save(save_path, 'jpeg')


# grayscale images
for i, row in tqdm(V.loc[V.channels == 1].iterrows()):
    # get image
    file_path = os.path.join(train_dir, row.directory, row.img_name)
    image = Image.open(file_path)
    
    # convert to RGB
    array = np.asarray(image, dtype='uint8')
    array = np.stack([array, array, array], axis=2)
    image = Image.fromarray(array)
    
    # save
    save_path = os.path.join(data_dir, 'val_no_resizing', str(row.category_number), row.img_name)
    image.save(save_path, 'jpeg')


train_size = len(T)
train_size


# RGB images
for i, row in tqdm(T.loc[T.channels == 3].iterrows()):
    # get image
    file_path = os.path.join(train_dir, row.directory, row.img_name)
    image = Image.open(file_path)
    
    # save
    save_path = os.path.join(data_dir, 'train_no_resizing', str(row.category_number), row.img_name)
    image.save(save_path, 'jpeg')

# grayscale images
for i, row in tqdm(T.loc[T.channels == 1].iterrows()):
    # get image
    file_path = os.path.join(train_dir, row.directory, row.img_name)
    image = Image.open(file_path)
    
    # convert to RGB
    array = np.asarray(image, dtype='uint8')
    array = np.stack([array, array, array], axis=2)
    image = Image.fromarray(array)
    
    # save
    save_path = os.path.join(data_dir, 'train_no_resizing', str(row.category_number), row.img_name)
    image.save(save_path, 'jpeg')



#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy import misc

import os
import shutil
from tqdm import tqdm 

# the folder from train.zip file
dir_train = 'f:\waste\train'

# remove non-images
os.remove(os.path.join(dir_train, '198.spider/RENAME2'))
shutil.rmtree(os.path.join(dir_train, '056.dog/greg'))
# we don't need the class with noise
shutil.rmtree(os.path.join(dir_train, '257.clutter'))


# # Collect metadata

subdirs = list(os.walk(dir_train))[1:]

# collect train metadata
train_metadata = []

for dir_path, _, files in tqdm(subdirs):
    
    dir_name = dir_path.split('/')[-1]
    
    for file_name in files:
        if not file_name.startswith('.'):
            # read image
            temp = misc.imread(os.path.join(dir_path, file_name)) 
            # collect image metadata
            image_metadata = []
            image_metadata.extend([dir_name, file_name])
            image_metadata.extend( 
                list(temp.shape) if len(temp.shape) == 3 
                else [temp.shape[0], temp.shape[1], 1]
            )
            image_metadata.extend([temp.nbytes, temp.dtype])
            # append image metadata to list
            train_metadata.append(image_metadata)


# # Explore metadata


M = pd.DataFrame(train_metadata)
M.columns = ['directory', 'img_name', 'height', 'width', 'channels', 'byte_size', 'bit_depth']

M['category_name'] = M.directory.apply(lambda x: x.split('.')[-1].lower())
M['img_extension'] = M.img_name.apply(lambda x: x.split('.')[-1])
M['category_number'] = M.directory.apply(lambda x: int(x.split('.')[0]))

# remove '101' from some category names
M.category_name = M.category_name.apply(lambda x: x[:-4] if '101' in x else x)


# number of grayscale images
(M.channels != 3).sum()


M.img_extension.unique()

M.bit_depth.unique()

# number of categories
M.category_name.nunique()


# # Create decoder


# class number -> class name
decode = {n: i for i, n in M.groupby('category_name').category_number.first().iteritems()}

np.save('decode.npy', decode)


# # Split data

# 20 images per class
V = M.groupby('category_name', group_keys=False).apply(lambda x: x.sample(n=20, replace=False))
V.sort_index(inplace=True)
M.drop(V.index, axis=0, inplace=True)

# train data
len(M)

# validation data
len(V)


# # Save split

M.to_csv('train_metadata.csv', index=False)

V.to_csv('val_metadata.csv', index=False)
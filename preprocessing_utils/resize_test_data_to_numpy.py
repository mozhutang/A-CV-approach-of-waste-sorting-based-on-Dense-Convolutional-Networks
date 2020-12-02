
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from PIL import Image
import torchvision.transforms as transforms
from scipy import misc

import os
from tqdm import tqdm 

# the folder from test.zip file
dir_test = '/home/ubuntu/test/'


# # Collect metadata

files = list(os.walk(dir_test))[0][2]
files

# collect test metadata
test_metadata = []
    
for file_name in tqdm(files):
    if not file_name.startswith('.'):
        # read image
        temp = misc.imread(os.path.join(dir_test, file_name)) 
        # collect image metadata
        image_metadata = [file_name]
        image_metadata.extend( 
            list(temp.shape) if len(temp.shape) == 3 
            else [temp.shape[0], temp.shape[1], 1]
        )
        image_metadata.extend([temp.nbytes, temp.dtype])
        # append image metadata to list
        test_metadata.append(image_metadata)


# # Explore metadata

M = pd.DataFrame(test_metadata)
M.columns = ['img_name', 'height', 'width', 'channels', 'byte_size', 'bit_depth']

M['img_extension'] = M['img_name'].apply(lambda x: x.split('.')[-1])

# number of grayscale images
(M.channels != 3).sum()

M.img_extension.unique()

M.bit_depth.unique()


# # Resize images

test_transform = transforms.Compose([
    transforms.Scale(224, Image.LANCZOS),
    transforms.CenterCrop(224)
])

n_images = len(M)
n_images


images = np.zeros((n_images, 224, 224, 3), dtype='uint8')
numbers = np.zeros((n_images,), dtype='int64')

# resize RGB images
for i, row in tqdm(M.loc[M.channels == 3].iterrows()):
    # get image
    file_path = os.path.join(dir_test, row.img_name)
    image = Image.open(file_path)
    
    # transform it
    image = test_transform(image)
    
    # convert to array
    array = np.asarray(image, dtype='uint8')
    
    images[i] = array
    numbers[i] = int(row.img_name[:-4])

# resize grayscale images
for i, row in tqdm(M.loc[M.channels == 1].iterrows()):
    file_path = os.path.join(dir_test, row.img_name)
    image = Image.open(file_path)
    
    # transform it
    image = test_transform(image)
    
    # convert to array
    array = np.asarray(image, dtype='uint8')
    
    # convert to RGB
    array = np.stack([array, array, array], axis=2)
    images[i] = array
    numbers[i] = int(row.img_name[:-4])


# # Show some resized images


# choose random images
to_show = np.random.choice(np.arange(0, n_images), size=2*4)


_, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 5))
axes = axes.flatten()
for i, j in enumerate(to_show):
    axes[i].set_axis_off();
    axes[i].imshow(images[j]/255);
    axes[i].set_title(numbers[j]);
plt.tight_layout()


# # Save

np.save('/home/ubuntu/data/test_images.npy', images) # size: 1.1 GB
np.save('/home/ubuntu/data/test_file_names.npy', numbers)



#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from PIL import Image, ImageEnhance
import torchvision.transforms as transforms

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

os.mkdir(data_dir + 'train')
for i in range(1, 256 + 1):
    os.mkdir(data_dir + 'train/' + str(i))


os.mkdir(data_dir + 'val')
for i in range(1, 256 + 1):
    os.mkdir(data_dir + 'val/' + str(i))


# # Resize val. images

val_transform = transforms.Compose([
    transforms.Scale(299, Image.LANCZOS),
    transforms.CenterCrop(299)
])

val_size = len(V)
val_size

# resize RGB images
for i, row in tqdm(V.loc[V.channels == 3].iterrows()):
    # get image
    file_path = os.path.join(train_dir, row.directory, row.img_name)
    image = Image.open(file_path)
    
    # transform it
    image = val_transform(image)
    
    # save
    save_path = os.path.join(data_dir, 'val', str(row.category_number), row.img_name)
    image.save(save_path, 'jpeg')

# resize grayscale images
for i, row in tqdm(V.loc[V.channels == 1].iterrows()):
    # get image
    file_path = os.path.join(train_dir, row.directory, row.img_name)
    image = Image.open(file_path)
    
    # transform it
    image = val_transform(image)
    
    # convert to RGB
    array = np.asarray(image, dtype='uint8')
    array = np.stack([array, array, array], axis=2)
    image = Image.fromarray(array)
    
    # save
    save_path = os.path.join(data_dir, 'val', str(row.category_number), row.img_name)
    image.save(save_path, 'jpeg')


# # Resize train images

enhancers = {
    0: lambda image, f: ImageEnhance.Color(image).enhance(f),
    1: lambda image, f: ImageEnhance.Contrast(image).enhance(f),
    2: lambda image, f: ImageEnhance.Brightness(image).enhance(f),
    3: lambda image, f: ImageEnhance.Sharpness(image).enhance(f)
}

factors = {
    0: lambda: np.random.uniform(0.4, 1.6),
    1: lambda: np.random.uniform(0.8, 1.2),
    2: lambda: np.random.uniform(0.8, 1.2),
    3: lambda: np.random.uniform(0.4, 1.6)
}

# randomly enhance images in random order
def enhance(image):
    order = [0, 1, 2, 3]
    np.random.shuffle(order)
    for i in order:
        f = factors[i]()
        image = enhancers[i](image, f)
    return image


train_transform_rare = transforms.Compose([
    transforms.Scale(384, Image.LANCZOS),
    transforms.RandomCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(enhance)
])

train_transform = transforms.Compose([
    transforms.Scale(384, Image.LANCZOS),
    transforms.RandomCrop(299),
    transforms.RandomHorizontalFlip(),
])


# number of images in each category
category_counts = dict(T.category_name.value_counts())
np.save('class_counts.npy', class_counts)

# sample with replacement 100 images from each category
T = T.groupby('category_name', group_keys=False).apply(lambda x: x.sample(n=100, replace=True))
T.reset_index(drop=True, inplace=True)


train_size = len(T)
train_size


# resize RGB images
for i, row in tqdm(T.loc[T.channels == 3].iterrows()):
    # get image
    file_path = os.path.join(train_dir, row.directory, row.img_name)
    image = Image.open(file_path)
    
    # transform it
    if category_counts[row.category_name] < 100:
        image = train_transform_rare(image)
    else:
        image = train_transform(image)
    
    # save
    new_image_name = str(i) + '_' + row.img_name
    save_path = os.path.join(data_dir, 'train', str(row.category_number), new_image_name)
    image.save(save_path, 'jpeg')


# resize grayscale images
for i, row in tqdm(T.loc[T.channels == 1].iterrows()):
    # get image
    file_path = os.path.join(train_dir, row.directory, row.img_name)
    image = Image.open(file_path)
    
    # transform it
    if category_counts[row.category_name] < 100:
        image = train_transform_rare(image)
    else:
        image = train_transform(image)
    
    # convert to RGB
    array = np.asarray(image, dtype='uint8')
    array = np.stack([array, array, array], axis=2)
    image = Image.fromarray(array)
    
    # save
    new_image_name = str(i) + '_' + row.img_name
    save_path = os.path.join(data_dir, 'train', str(row.category_number), new_image_name)
    image.save(save_path, 'jpeg')


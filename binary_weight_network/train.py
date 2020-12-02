
#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import numpy as np
from math import ceil
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import sys
sys.path.append('../training_utils/')
from data_utils import get_folders, get_class_weights

# utils specific to quantization
from utils import train, quantize
torch.cuda.is_available()

torch.backends.cudnn.benchmark = True

# # Create data iterators

batch_size = 32

train_folder, val_folder = get_folders()

train_iterator = DataLoader(
    train_folder, batch_size=batch_size, num_workers=4,
    shuffle=True, pin_memory=True
)

val_iterator = DataLoader(
    val_folder, batch_size=64, num_workers=4,
    shuffle=False, pin_memory=True
)

# number of training samples
train_size = len(train_folder.imgs)
train_size

from get_densenet import get_model


# w[j]: 1/number_of_samples_in_class_j
# decode: folder name to class name (in human readable format)
w, decode = get_class_weights(val_folder.class_to_idx)


model, criterion, optimizer = get_model(class_weights=torch.FloatTensor(w/w.sum()))

# load pretrained model, accuracy ~85%
model.load_state_dict(torch.load('../densenet/model121.pytorch_state'))


# #### Keep copy of full precision kernels


# copy all full precision kernels of the model
all_fp_kernels = [
    Variable(kernel.data.clone(), requires_grad=True) 
    for kernel in optimizer.param_groups[2]['params']
]
# all_fp_kernels - kernel tensors of all convolutional layers 
# (with the exception of the first conv layer)


# #### initial quantization 

# these kernels will be quantized
all_kernels = [kernel for kernel in optimizer.param_groups[2]['params']]


for k, k_fp in zip(all_kernels, all_fp_kernels):
    
    k.data = quantize(k_fp.data)


# #### parameter updaters


# optimizer for updating only all_fp_kernels
optimizer_fp = optim.SGD(all_fp_kernels, lr=1e-3, momentum=0.9, nesterov=True)


# # Train


n_epochs = 15
n_batches = ceil(train_size/batch_size)
n_batches


get_ipython().run_cell_magic('time', '', 'all_losses, _ = train(\n    model, criterion, \n    optimizer, optimizer_fp,\n    train_iterator, n_epochs, n_batches,\n    val_iterator, validation_step=531, n_validation_batches=80\n)\n# epoch logloss    accuracy     top5_accuracy time  (first value: train, second value: val)')


# # Loss/epoch plots


epochs = [x[0] for x in all_losses]
plt.plot(epochs, [x[1] for x in all_losses], label='train');
plt.plot(epochs, [x[2] for x in all_losses], label='val');
plt.legend();
plt.xlabel('epoch');
plt.ylabel('loss');


plt.plot(epochs, [x[3] for x in all_losses], label='train');
plt.plot(epochs, [x[4] for x in all_losses], label='val');
plt.legend();
plt.xlabel('epoch');
plt.ylabel('accuracy');


plt.plot(epochs, [x[5] for x in all_losses], label='train');
plt.plot(epochs, [x[6] for x in all_losses], label='val');
plt.legend();
plt.xlabel('epoch');
plt.ylabel('top5_accuracy');


# # Save

model.cpu();
torch.save(model.state_dict(), 'model_binary_quantization.pytorch_state')


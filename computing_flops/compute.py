
#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
sys.path.append('../resnet/')
sys.path.append('../densenet/')
sys.path.append('../squeezenet/')

from get_resnet18 import get_model as get_resnet
from get_densenet121 import get_model as get_densenet
from get_squeezenet import get_model as get_squeezenet

resnet, _, _ = get_resnet()
densenet, _, _ = get_densenet()
squeezenet, _, _ = get_squeezenet()

def count_flops(model, input_image_size):
    
    # flops counts from each layer
    counts = []
    
    # loop over all model parts
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            def hook(module, input):
                factor = 2*module.in_channels*module.out_channels
                factor *= module.kernel_size[0]*module.kernel_size[1]
                factor //= module.stride[0]*module.stride[1]
                counts.append(
                    factor*input[0].data.shape[2]*input[0].data.shape[3]
                )
            m.register_forward_pre_hook(hook)
        elif isinstance(m, nn.Linear):
            counts += [
                2*m.in_features*m.out_features
            ]
        
    noise_image = torch.rand(
        1, 3, input_image_size, input_image_size
    )
    # one forward pass
    _ = model(Variable(noise_image.cuda(), volatile=True))
    return sum(counts)


input_image_size = 299

count_flops(resnet, input_image_size)

count_flops(densenet, input_image_size)

count_flops(squeezenet, input_image_size)


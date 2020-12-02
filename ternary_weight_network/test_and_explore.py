
#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
from math import ceil
from sklearn.metrics import accuracy_score, log_loss
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import sys
sys.path.append('../training_utils/')
from data_utils import get_folders, get_class_weights
from train_utils import predict
from diagnostic_tools import top_k_accuracy, per_class_accuracy,    count_params, entropy, model_calibration, show_errors, most_confused_classes,    most_inaccurate_k_classes

torch.cuda.is_available()


# # Load the model


from get_densenet import get_model


_, val_folder = get_folders()

# w[j]: 1/number_of_samples_in_class_j
# decode: folder name to class name (in human readable format)
w, decode = get_class_weights(val_folder.class_to_idx)


model, _, _ = get_model(class_weights=torch.FloatTensor(w/w.sum()))

# load pretrained quantized model
model.load_state_dict(torch.load('model_ternary_quantization.pytorch_state'))


# number of params in the model
count_params(model)


# # Show some quantized kernel tensors (there are 119 such kernels overall)


# all quantized kernels
all_kernels = [
    (n, p.data) for n, p in model.named_parameters() 
    if ('denseblock' in n or 'transition' in n) and 'conv' in n
]


# show distribution of weights in kernels that are in the beginning of the network 
_, axes = plt.subplots(nrows=6, ncols=4, figsize=(14, 10))
axes = axes.flatten()
for i, (name, kernel) in enumerate(all_kernels[:24]):
    axes[i].hist(kernel.cpu().numpy().reshape(-1));
    axes[i].set_title(name[9:-7]);

plt.tight_layout()


# kernels in the end of the network 
_, axes = plt.subplots(nrows=6, ncols=4, figsize=(14, 10))
axes = axes.flatten()
for i, (name, kernel) in enumerate(all_kernels[-24:]):
    axes[i].hist(kernel.cpu().numpy().reshape(-1));
    axes[i].set_title(name[9:-7]);

plt.tight_layout()


# # sparcity distribution


sparcity = []
for n, p in all_kernels:
    sparcity.append(p.eq(0.0).float().mean())


plt.hist(sparcity);


np.mean(sparcity)


# # scaling factors


upper_scaling_factor = []
lower_scaling_factor = []
for n, p in all_kernels:
    upper_scaling_factor.append(p.max())
    lower_scaling_factor.append(p.min())


plt.hist(upper_scaling_factor);
plt.hist(lower_scaling_factor);


# # Error analysis

# ### get human readable class names

# index to class name
decode = {val_folder.class_to_idx[k]: decode[int(k)] for k in val_folder.class_to_idx}


# ### get all predictions and all misclassified images 

val_iterator_no_shuffle = DataLoader(
    val_folder, batch_size=64, shuffle=False
)


val_predictions, val_true_targets,    erroneous_samples, erroneous_targets,    erroneous_predictions = predict(model, val_iterator_no_shuffle, return_erroneous=True)
# erroneous_samples: images that were misclassified
# erroneous_targets: their true labels
# erroneous_predictions: predictions for them


# ### number of misclassified images (there are overall 5120 images in the val dataset)

n_errors = len(erroneous_targets)
n_errors


# ### logloss and accuracies


log_loss(val_true_targets, val_predictions)

accuracy_score(val_true_targets, val_predictions.argmax(1))

print(top_k_accuracy(val_true_targets, val_predictions, k=(2, 3, 4, 5, 10)))


# ### entropy of predictions

hits = val_predictions.argmax(1) == val_true_targets

plt.hist(
    entropy(val_predictions[hits]), bins=30, 
    normed=True, alpha=0.7, label='correct prediction'
);
plt.hist(
    entropy(val_predictions[~hits]), bins=30, 
    normed=True, alpha=0.5, label='misclassification'
);
plt.legend();
plt.xlabel('entropy of predictions');


# ### confidence of predictions

plt.hist(
    val_predictions[hits].max(1), bins=30, 
    normed=True, alpha=0.7, label='correct prediction'
);
plt.hist(
    val_predictions[~hits].max(1), bins=30, 
    normed=True, alpha=0.5, label='misclassification'
);
plt.legend();
plt.xlabel('confidence of predictions');


# ### difference between biggest and second biggest probability

sorted_correct = np.sort(val_predictions[hits], 1)
sorted_incorrect = np.sort(val_predictions[~hits], 1)

plt.hist(
    sorted_correct[:, -1] - sorted_correct[:, -2], bins=30, 
    normed=True, alpha=0.7, label='correct prediction'
);
plt.hist(
    sorted_incorrect[:, -1] - sorted_incorrect[:, -2], bins=30, 
    normed=True, alpha=0.5, label='misclassification'
);
plt.legend();
plt.xlabel('difference');


# ### probabilistic calibration of the model

model_calibration(val_true_targets, val_predictions, n_bins=10)


# ### per class accuracies


per_class_acc = per_class_accuracy(val_true_targets, val_predictions)
plt.hist(per_class_acc);
plt.xlabel('accuracy');


most_inaccurate_k_classes(per_class_acc, 15, decode)


# ### class accuracy vs. number of samples in the class


plt.scatter((1.0/w), per_class_acc);
plt.ylabel('class accuracy');
plt.xlabel('number of available samples');


# ### most confused pairs of classes

confused_pairs = most_confused_classes(
    val_true_targets, val_predictions, decode, min_n_confusions=4
)
confused_pairs


# ### show some low entropy errors


erroneous_entropy = entropy(erroneous_predictions)
mean_entropy = erroneous_entropy.mean()
low_entropy = mean_entropy < erroneous_entropy
mean_entropy

show_errors(
    erroneous_samples[low_entropy], 
    erroneous_predictions[low_entropy], 
    erroneous_targets[low_entropy], 
    decode
)


# ### show some high entropy errors


show_errors(
    erroneous_samples[~low_entropy], 
    erroneous_predictions[~low_entropy], 
    erroneous_targets[~low_entropy], 
    decode
)


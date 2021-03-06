import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
sys.path.append('../training_utils/')
from train_utils import evaluate, accuracy


# hyperparameter
t = 0.1


def initial_scales(kernel):
    return 1.0, 1.0


def quantize(kernel, w_p, w_n):
    delta = t*kernel.abs().max()
    a = (kernel > delta).float()
    b = (kernel < -delta).float()
    return w_p*a + (-w_n*b)


def get_grads(kernel_grad, kernel, w_p, w_n):
    delta = t*kernel.abs().max()
    # masks
    a = (kernel > delta).float()
    b = (kernel < -delta).float()
    c = torch.ones(kernel.size()).cuda() - a - b
    # scaled kernel grad and grad for scaling factors (w_p, w_n)
    return w_p*a*kernel_grad + w_n*b*kernel_grad + 1.0*c*kernel_grad,\
        (a*kernel_grad).sum(), (b*kernel_grad).sum()


def optimization_step(model, criterion,
                      optimizer, optimizer_fp, optimizer_sf,
                      x_batch, y_batch):

    x_batch, y_batch = Variable(x_batch.cuda()), Variable(y_batch.cuda(async=True))
    # use quantized model
    logits = model(x_batch)

    # compute logloss
    loss = criterion(logits, y_batch)
    batch_loss = loss.data[0]

    # compute accuracies
    pred = F.softmax(logits)
    batch_accuracy, batch_top5_accuracy = accuracy(y_batch, pred, top_k=(1, 5))

    optimizer.zero_grad()
    optimizer_fp.zero_grad()
    optimizer_sf.zero_grad()
    # compute grads for quantized model
    loss.backward()

    all_kernels = optimizer.param_groups[2]['params']
    all_fp_kernels = optimizer_fp.param_groups[0]['params']
    scaling_factors = optimizer_sf.param_groups[0]['params']

    for i in range(len(all_kernels)):

        # get quantized kernel
        k = all_kernels[i]

        # get corresponding full precision kernel
        k_fp = all_fp_kernels[i]

        # get scaling factors for quantized kernel
        f = scaling_factors[i]
        w_p, w_n = f.data[0], f.data[1]

        # get modified grads
        k_fp_grad, w_p_grad, w_n_grad = get_grads(k.grad.data, k.data, w_p, w_n)
        # WARNING: this is not like in the original paper.
        # In the original paper: k.data -> k_fp.data

        # grad for full precision kernel
        k_fp.grad = Variable(k_fp_grad)

        # we don't need to update quantized kernel directly
        k.grad.data.zero_()

        # grad for scaling factors
        f.grad = Variable(torch.FloatTensor([w_p_grad, w_n_grad]).cuda())

    # update the last fc layer and all batch norm params in quantized model
    optimizer.step()

    # update full precision kernels
    optimizer_fp.step()

    # update scaling factors
    optimizer_sf.step()

    # update quantized kernels
    for i in range(len(all_kernels)):

        k = all_kernels[i]
        k_fp = all_fp_kernels[i]
        f = scaling_factors[i]
        w_p, w_n = f.data[0], f.data[1]

        k.data = quantize(k_fp.data, w_p, w_n)

    return batch_loss, batch_accuracy, batch_top5_accuracy


# just training helper, nothing special
def train(model, criterion,
          optimizer, optimizer_fp, optimizer_sf,
          train_iterator, n_epochs, n_batches,
          val_iterator, validation_step, n_validation_batches,
          saving_step=None, lr_scheduler=None):

    all_losses = []
    all_models = []

    is_reduce_on_plateau = isinstance(lr_scheduler, ReduceLROnPlateau)

    running_loss = 0.0
    running_accuracy = 0.0
    running_top5_accuracy = 0.0
    start = time.time()
    model.train()

    for epoch in range(0, n_epochs):
        for step, (x_batch, y_batch) in enumerate(train_iterator, 1 + epoch*n_batches):

            if lr_scheduler is not None and not is_reduce_on_plateau:
                optimizer = lr_scheduler(optimizer, step)

            batch_loss, batch_accuracy, batch_top5_accuracy = optimization_step(
                model, criterion,
                optimizer, optimizer_fp, optimizer_sf,
                x_batch, y_batch
            )
            running_loss += batch_loss
            running_accuracy += batch_accuracy
            running_top5_accuracy += batch_top5_accuracy

            if step % validation_step == 0:
                model.eval()
                test_loss, test_accuracy, test_top5_accuracy = evaluate(
                    model, criterion, val_iterator, n_validation_batches
                )
                end = time.time()

                print('{0:.2f}  {1:.3f} {2:.3f}  {3:.3f} {4:.3f}  {5:.3f} {6:.3f}  {7:.3f}'.format(
                    step/n_batches, running_loss/validation_step, test_loss,
                    running_accuracy/validation_step, test_accuracy,
                    running_top5_accuracy/validation_step, test_top5_accuracy,
                    end - start
                ))
                all_losses += [(
                    step/n_batches,
                    running_loss/validation_step, test_loss,
                    running_accuracy/validation_step, test_accuracy,
                    running_top5_accuracy/validation_step, test_top5_accuracy
                )]

                if is_reduce_on_plateau:
                    lr_scheduler.step(test_accuracy)

                running_loss = 0.0
                running_accuracy = 0.0
                running_top5_accuracy = 0.0
                start = time.time()
                model.train()

            if saving_step is not None and step % saving_step == 0:

                print('saving')
                model.cpu()
                clone = copy.deepcopy(model)
                all_models += [clone.state_dict()]
                model.cuda()

    return all_losses, all_models

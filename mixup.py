import torch
import numpy as np

def mixup_data(x_1, x_2, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x_1.size()[0]
    mixed_x = lam * x_1 + (1 - lam) * x_2

    return mixed_x, lam




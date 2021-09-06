from torch.autograd import Variable
import torch

import numpy as np

def loss_backprop(generator, criterion, out, targets, normalize, bp=True):
    """
    Memory optmization. Compute each timestep separately and sum grads.
    """
    assert out.size(1) == targets.size(1)
    total = 0.0
    out_grad = []
    
    for i in range(out.size(1)):
        out_column = Variable(out[:, i].data, requires_grad=True)
        gen = generator(out_column)
        loss = criterion(gen, targets[:, i]) / normalize
        total += loss.data
        loss.backward()
        out_grad.append(out_column.grad.data.clone())
    if bp:
        out_grad = torch.stack(out_grad, dim=1)
        out.backward(gradient=out_grad)

    return total

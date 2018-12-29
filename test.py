'''
I transformed this from https://github.com/batzner/indrnn/blob/master/examples/addition_rnn.py to test functionality
'''

import numpy as np
import torch
from torch.autograd import Variable
from indrnn import IndRNNCell


def testIndRNNCell():
    x = Variable(torch.Tensor([[1., 1., 1., 1.]]))
    m = Variable(torch.Tensor([[2., 2., 2., 2.]]))
    recurrent_init = torch.Tensor([-5., -2., 0.1, 5.])
    cell = IndRNNCell(4, 4, hidden_min_abs=1., hidden_max_abs=3.)
    cell.weight_ih.data.fill_(1)
    cell.weight_hh.data.copy_(recurrent_init)
    cell.check_bounds()
    output = cell(x, m)

    # Recurrent Weights u should be -3, -2, 1, 3
    # Pre-activations (4 + 2*u) should be -2, 0, 6, 10
    np.testing.assert_array_equal(output.data.numpy(), [[0., 0., 6., 10.]])


testIndRNNCell()

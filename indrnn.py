import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class IndRNNCell(nn.Module):
    r"""An IndRNN cell with tanh or ReLU non-linearity.

    .. math::

        h' = \tanh(w_{ih} * x + b_{ih}  +  w_{hh} (*) h)
    With (*) being element-wise vector multiplication.
    If nonlinearity='relu', then ReLU is used in place of tanh.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If ``False``, then the layer does not use bias weights b_ih and b_hh.
            Default: ``True``
        nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'relu'
        hidden_min_abs: Minimal absolute inital value for hidden weights. Default: 0
        hidden_max_abs: Maximal absolute inital value for hidden weights. Default: None

    Inputs: input, hidden
        - **input** (batch, input_size): tensor containing input features
        - **hidden** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.

    Outputs: h'
        - **h'** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(input_size x hidden_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(hidden_size)`

    Examples::

        >>> rnn = nn.IndRNNCell(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> hx = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="relu",
                 hidden_min_abs=0, hidden_max_abs=None):
        super(IndRNNCell, self).__init__()
        self.hidden_max_abs = hidden_max_abs
        self.hidden_min_abs = hidden_min_abs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        if self.nonlinearity == "tanh":
            self.activation = F.tanh
        elif self.nonlinearity == "relu":
            self.activation = F.relu
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if "bias" in name:
                weight.data.zero_()
            elif "weight_hh" in name:
                if self.hidden_max_abs:
                    stdv_ = self.hidden_max_abs
                else:
                    stdv_ = stdv
                weight.data.uniform_(-stdv_, stdv_)
            elif "weight_ih" in name:
                weight.data.normal_(0, 0.01)
            else:
                weight.data.normal_(0, 0.01)
                # weight.data.uniform_(-stdv, stdv)
        self.check_bounds()

    def check_bounds(self):
        if self.hidden_min_abs:
            abs_kernel = torch.abs(self.weight_hh.data)
            min_abs_kernel = torch.clamp(abs_kernel, min=self.hidden_min_abs)
            self.weight_hh.data.copy_(
                torch.mul(torch.sign(self.weight_hh.data), min_abs_kernel))
        if self.hidden_max_abs:
            self.weight_hh.data.copy_(
                torch.clamp(self.weight_hh.data, max=self.hidden_max_abs,
                            min=-self.hidden_max_abs))

    def forward(self, input, hx):
        return self.activation(F.linear(
            input, self.weight_ih, self.bias_ih) + F.mul(self.weight_hh, hx))


class IndRNN(nn.Module):
    r"""Applies a multi-layer IndRNN with `tanh` or `ReLU` non-linearity to an
    input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

        h_t = \tanh(w_{ih} x_t + b_{ih}  +  w_{hh} (*) h_{(t-1)})

    where :math:`h_t` is the hidden state at time `t`, and :math:`x_t` is
    the hidden state of the previous layer at time `t` or :math:`input_t`
    for the first layer. (*) is element-wise multiplication.
    If :attr:`nonlinearity`='relu', then `ReLU` is used instead of `tanh`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers.
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_norm: If ``True``, then batch normalization is applied after each time step
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)`

    Inputs: input, h_0
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          or :func:`torch.nn.utils.rnn.pack_sequence`
          for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: output, h_n
        - **output** of shape `(seq_len, batch, hidden_size * num_directions)`: tensor
          containing the output features (`h_k`) from the last layer of the RNN,
          for each `k`.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
          been given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for `k = seq_len`.

    Attributes:
        cells[k]: individual IndRNNCells containing the weights

    Examples::

        >>> rnn = nn.IndRNN(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output = rnn(input, h0)
    """

    def __init__(self, input_size, hidden_size, n_layer=1, batch_norm=False,
                 batch_first=False, bidirectional=False, **kwargs):
        super(IndRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_norm = batch_norm
        self.n_layer = n_layer
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        num_directions = 2 if self.bidirectional else 1

        if batch_first:
            self.time_index = 1
            self.batch_index = 0
        else:
            self.time_index = 0
            self.batch_index = 1

        cells = []
        for i in range(n_layer):
            directions = []
            for dir in range(num_directions):
                if i == 0:
                    directions.append(IndRNNCell(
                        input_size, hidden_size, **kwargs))
                else:
                    directions.append(IndRNNCell(
                        hidden_size * num_directions, hidden_size, **kwargs))
            cells.append(nn.ModuleList(directions))
        self.cells = nn.ModuleList(cells)

        if batch_norm:
            bns = []
            for i in range(n_layer):
                directions = []
                for dir in range(num_directions):
                    directions.append(nn.BatchNorm1d(hidden_size))
                bns.append(nn.ModuleList(directions))
            self.bns = nn.ModuleList(bns)

        h0 = torch.zeros(hidden_size * num_directions)
        self.register_buffer('h0', torch.autograd.Variable(h0))

    def forward(self, x, hidden=None):
        batch_norm = self.batch_norm
        time_index = self.time_index
        batch_index = self.batch_index
        num_directions = 2 if self.bidirectional else 1

        for i, directions in enumerate(self.cells):
            hx = self.h0.unsqueeze(0).expand(
                x.size(batch_index),
                self.hidden_size * num_directions).contiguous()

            x_n = []
            for dir, cell in enumerate(directions):
                hx_cell = hx[
                    :, self.hidden_size * dir: self.hidden_size * (dir + 1)]
                cell.check_bounds()
                outputs = []
                hiddens = []
                x_T = torch.unbind(x, time_index)
                if dir == 1:
                    x_T = reversed(x_T)
                for x_t in x_T:
                    hx_cell = cell(x_t, hx_cell)
                    if batch_norm:
                        hx_cell = self.bns[i][dir](hx_cell)
                    outputs.append(hx_cell)
                if dir == 1:
                    outputs = outputs[::-1]
                x_cell = torch.stack(outputs, time_index)
                x_n.append(x_cell)
                hiddens.append(hx_cell)
            x = torch.cat(x_n, -1)
        return x.squeeze(2), torch.cat(hiddens, -1)

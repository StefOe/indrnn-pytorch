"""Module using IndRNNCell to solve the addition problem
The addition problem is stated in https://arxiv.org/abs/1803.04831. The
hyper-parameters are taken from that paper as well. The network should
converge to a MSE around zero after 1500-3000 steps.

I transformed this from https://github.com/batzner/indrnn/blob/master/examples/addition_rnn.py

"""
from indrnn import IndRNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


# Parameters taken from https://arxiv.org/abs/1803.04831
TIME_STEPS = 100
NUM_UNITS = 128
LEARNING_RATE = 0.0002
NUM_LAYERS = 2
BATCH_NORM = False
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)

# Parameters taken from https://arxiv.org/abs/1511.06464
BATCH_SIZE = 50

cuda = torch.cuda.is_available()


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer=2):
        super(Net, self).__init__()
        self.indrnn = IndRNN(
            input_size, hidden_size, n_layer, batch_norm=BATCH_NORM,
            cuda=cuda, hidden_max_abs=RECURRENT_MAX, step_size=TIME_STEPS)
        self.lin = nn.Linear(hidden_size, 1)
        self.lin.bias.data.fill_(.1)
        self.lin.weight.data.normal_(0, .01)

    def forward(self, x, hidden=None):
        y = self.indrnn(x, hidden)
        return self.lin(y[:, -1]).squeeze(1)


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.cell1 = nn.LSTM(2, NUM_UNITS)
        self.lin = nn.Linear(NUM_UNITS, 1)

    def forward(self, x, hidden=None):
        x, hidden = self.cell1(x, hidden)
        return self.lin(x[:, -1]).squeeze(1)


def main():
    # build model
    model = Net(2, NUM_UNITS, NUM_LAYERS)
    # model = LSTM()
    if cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    model.train()
    step = 0
    while True:
        losses = []
        for _ in range(100):
            # Generate new input data
            data, target = get_batch()
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            model.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu()[0])
            step += 1

        print(
            "Step [x100] {} MSE {}".format(int(step / 100), np.mean(losses)))


def get_batch():
    """Generate the adding problem dataset"""
    # Build the first sequence
    add_values = np.random.rand(BATCH_SIZE, TIME_STEPS)

    # Build the second sequence with one 1 in each half and 0s otherwise
    add_indices = np.zeros_like(add_values)
    half = int(TIME_STEPS / 2)
    for i in range(BATCH_SIZE):
        first_half = np.random.randint(half)
        second_half = np.random.randint(half, TIME_STEPS)
        add_indices[i, [first_half, second_half]] = 1

    # Zip the values and indices in a third dimension:
    # inputs has the shape (batch_size, time_steps, 2)
    inputs = np.dstack((add_values, add_indices))
    targets = np.sum(np.multiply(add_values, add_indices), axis=1)
    return torch.from_numpy(inputs).float(), torch.from_numpy(targets).float()


if __name__ == "__main__":
    main()

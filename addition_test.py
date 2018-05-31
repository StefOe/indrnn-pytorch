"""Module using IndRNNCell to solve the addition problem
The addition problem is stated in https://arxiv.org/abs/1803.04831. The
hyper-parameters are taken from that paper as well. The network should
converge to a MSE around zero after 1500-3000 steps.

"""
from indrnn import IndRNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='PyTorch IndRNN Addition test')
# Default parameters taken from https://arxiv.org/abs/1803.04831
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate (default: 0.0002)')
parser.add_argument('--time-steps', type=int, default=100,
                    help='length of addition problem (default: 100)')
parser.add_argument('--n-layer', type=int, default=2,
                    help='number of layer of IndRNN (default: 2)')
parser.add_argument('--hidden_size', type=int, default=128,
                    help='number of hidden units in one IndRNN layer(default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--batch-norm', action='store_true', default=False,
                    help='enable frame-wise batch normalization after each layer')
parser.add_argument('--log-interval', type=int, default=100,
                    help='after how many iterations to report performance')
parser.add_argument('--model', type=str, default="IndRNN",
                    help='if either IndRNN or LSTM cells should be used for optimization')


# Default parameters taken from https://arxiv.org/abs/1511.06464
parser.add_argument('--batch-size', type=int, default=50,
                    help='input batch size for training (default: 50)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

RECURRENT_MAX = pow(2, 1 / args.time_steps)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer=2):
        super(Net, self).__init__()
        self.indrnn = IndRNN(
            input_size, hidden_size, n_layer, batch_norm=args.batch_norm,
            hidden_max_abs=RECURRENT_MAX)
        self.lin = nn.Linear(hidden_size, 1)
        self.lin.bias.data.fill_(.1)
        self.lin.weight.data.normal_(0, .01)

    def forward(self, x, hidden=None):
        y = self.indrnn(x, hidden)
        return self.lin(y[-1]).squeeze(1)


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.cell1 = nn.LSTM(2, args.hidden_size)
        self.lin = nn.Linear(args.hidden_size, 1)

    def forward(self, x, hidden=None):
        x, hidden = self.cell1(x, hidden)
        return self.lin(x[-1]).squeeze(1)


def main():
    # build model
    if args.model == "IndRNN":
        model = Net(2, args.hidden_size, args.n_layer)
    elif args.model == "LSTM":
        model = LSTM()
    else:
        raise Exception("unsupported cell model")
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    model.train()
    step = 0
    while True:
        losses = []
        for _ in range(args.log_interval):
            # Generate new input data
            data, target = get_batch()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            model.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu().item())
            step += 1

        print(
            "MSE after {} iterations: {}".format(step, np.mean(losses)))

def get_batch():
    """Generate the adding problem dataset"""
    # Build the first sequence
    add_values = torch.rand(args.time_steps, args.batch_size)

    # Build the second sequence with one 1 in each half and 0s otherwise
    add_indices = torch.zeros_like(add_values)
    half = int(args.time_steps / 2)
    for i in range(args.batch_size):
        first_half = np.random.randint(half)
        second_half = np.random.randint(half, args.time_steps)
        add_indices[first_half, i] = 1
        add_indices[second_half, i] = 1

    # Zip the values and indices in a third dimension:
    # inputs has the shape (time_steps, batch_size, 2)
    inputs = torch.stack((add_values, add_indices), dim=-1)
    targets = torch.mul(add_values, add_indices).sum(dim=0)
    return inputs, targets

if __name__ == "__main__":
    main()

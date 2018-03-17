"""Module using IndRNNCell to solve the sequential MNIST task.
The hyper-parameters are taken from that paper as well.

"""
from indrnn import IndRNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np


# Parameters taken from https://arxiv.org/abs/1803.04831
TIME_STEPS = 784
NUM_UNITS = 128
LEARNING_RATE = 2e-4
NUM_LAYERS = 6
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
BATCH_NORM = True

# Parameters taken from https://arxiv.org/pdf/1511.06464
BATCH_SIZE = 256
MAX_STEPS = 10000 # unsure on this one

cuda = torch.cuda.is_available()


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer=2):
        super(Net, self).__init__()
        self.indrnn = IndRNN(
            input_size, hidden_size, n_layer, batch_norm=BATCH_NORM,
            cuda=cuda, hidden_max_abs=RECURRENT_MAX, step_size=TIME_STEPS)
        self.lin = nn.Linear(hidden_size, 10)
        self.lin.bias.data.fill_(.1)
        self.lin.weight.data.normal_(0, .01)

    def forward(self, x, hidden=None):
        y = self.indrnn(x, hidden)
        return self.lin(y[:, -1]).squeeze(1)


def main():
    # build model
    model = Net(1, NUM_UNITS, NUM_LAYERS)
    # model = LSTM()
    if cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # load data
    train_data, test_data = sequential_MNIST(BATCH_SIZE, cuda=cuda)

    # Train the model
    model.train()
    step = 0
    epochs = 0
    while step < MAX_STEPS:
        losses = []
        for data, target in train_data:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            model.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu()[0])
            step += 1
            if step >= MAX_STEPS:
                break
        epochs += 1
        print(
            "Epoch {} cross_entropy {}".format(
                epochs, np.mean(losses)))

    # get test error
    model.eval()
    correct = 0
    for data, target in test_data:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        out = model(data)
        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print(
        "Test accuracy:: {:.4f}".format(
            100. * correct / len(test_data.dataset)))




def sequential_MNIST(batch_size, cuda=False, dataset_folder='./data'):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dataset_folder, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                           # transform to sequence
                           transforms.Lambda(lambda x: x.view(-1, 1))
                       ])),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dataset_folder, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            # transform to sequence
            transforms.Lambda(lambda x: x.view(-1, 1))
        ])),
        batch_size=batch_size, shuffle=False, **kwargs)

    return (train_loader, test_loader)


if __name__ == "__main__":
    main()

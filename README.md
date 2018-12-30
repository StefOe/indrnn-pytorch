# indrnn-pytorch
pytorch implementation of the IndRNN Paper (https://arxiv.org/pdf/1803.04831.pdf)

The test functions are adapted from the tensorflow implementation (https://github.com/batzner/indrnn) and the theano implementation (https://github.com/Sunnydreamrain/IndRNN_Theano_Lasagne).
Tested with Python3.7 and pytorch 1.0.0

IndRNNv2 version should be faster with GPUs, especially for bidirectional networks.
Seconds per 100 iterations with GPU-P100 on the addition test and batch size of 50:

| IndRNN | IndRNNv2 |
| -----: | -------: |
|    6.7 |     3.65 |

Seconds per epoch with GPU-P100 on SeqMNIST and batch size of 256:

| IndRNN | IndRNNv2 |
| -----: | -------: |
|    394 |      114 |

TODOs:
-get parameters for MNIST experiments
-add permutation MNIST test

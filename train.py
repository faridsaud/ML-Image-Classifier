import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='Train a neural network to identify flowers.')

parser.add_argument("data_dir", help="directory of the  data")

parser.add_argument("--save_dir", help="directory where the checkpoints will be saved")

parser.add_argument("--arch", help="architecture of the pre-trained network",  choices=['resnet18', 'alexnet', 'vgg16', 'squeezenet', 'densenet', 'inception',
                                                                                      'googlenet', 'shufflenet', 'mobilenet', 'resnext50_32x4d'])

parser.add_argument("--learning_rate", type=float, help="learning rate")

parser.add_argument("--hidden_units ", type=int, help="hidden units for each hidden layer", nargs='+')

parser.add_argument("--epochs", type=int, help="epochs")

parser.add_argument("--gpu", help="use cuda instead of cpu")

# additional parameters
parser.add_argument("--input", type=int, help="number of input nodes", default=1024)
parser.add_argument("--output", type=int, help="number of output nodes", default=102)


args = parser.parse_args()






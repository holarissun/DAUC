'''load_data.py'''
import ddu_dirty_mnist
import argparse
import torch
import torchvision
from torch.autograd import Variable
import torch.utils.data.dataloader as Data
import numpy as np
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--inserted_ood_k",default=0, type=int)
parser.add_argument("--gpu_idx",default=None, type=int)
parser.add_argument("--batch_size",default=64, type=int)
parser.add_argument("--repeat",default=0, type=int)
parser.add_argument("--epochs",default=15, type=int)
parser.add_argument("--max_epochs",default=50, type=int)
parser.add_argument("--kernel",default='gaussian', type=str)
parser.add_argument("--bandwidth",default=1.0, type=float)
parser.add_argument("--matrix_mode",default='valid',type=str)
parser.add_argument("--validation_split",default=0.85,type=float)
args = parser.parse_args()
if args.gpu_idx is not None:
    torch.cuda.set_device(args.gpu_idx)
else:
    torch.cuda.set_device(int(args.repeat%8)) # auto-alloc gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('data/DMNIST/', exist_ok=True)

dirty_mnist_train = ddu_dirty_mnist.DirtyMNIST(".", train=True, download=True, device="cuda")
dirty_mnist_test = ddu_dirty_mnist.DirtyMNIST(".", train=False, download=True, device="cuda")
len(dirty_mnist_train), len(dirty_mnist_test)

train_dirty_arr = []
test_dirty_arr = []
train_dirty_label = []
test_dirty_label = []

for i in dirty_mnist_train:
    train_dirty_arr.append(i[0][0].cpu().numpy())
    train_dirty_label.append(i[1].cpu().numpy())
import numpy as np
train_dirty_arr = np.asarray(train_dirty_arr)
train_dirty_label = np.asarray(train_dirty_label)

for i in dirty_mnist_test:
    test_dirty_arr.append(i[0][0].cpu().numpy())
    test_dirty_label.append(i[1].cpu().numpy())

    
test_dirty_arr = np.asarray(test_dirty_arr)
test_dirty_label = np.asarray(test_dirty_label)

train_arr = np.asarray(train_dirty_arr[train_dirty_label!=args.inserted_ood_k])
train_label = np.asarray(train_dirty_label[train_dirty_label!=args.inserted_ood_k])
valid_idx = sorted(np.random.choice(len(test_dirty_arr), int(args.validation_split * len(test_dirty_arr) ), replace = False))
test_idx = sorted(list(set([_ for _ in range(len(test_dirty_arr))]) - set(valid_idx)))

test_boundary_idx = np.asarray(test_idx) >= 10000 
valid_dirty_boundary = np.asarray(valid_idx) >= 10000 
# the first 10k data are from MNIST, and the following 60k are from Ambiguous MNIST
# reference:
# https://arxiv.org/pdf/2102.11582.pdf
# https://blackhc.github.io/ddu_dirty_mnist/

valid_arr = test_dirty_arr[valid_idx]
valid_label = test_dirty_label[valid_idx]

valid_dirty_boundary = valid_dirty_boundary[valid_label != args.inserted_ood_k]
valid_arr = valid_arr[valid_label != args.inserted_ood_k]
valid_label = valid_label[valid_label != args.inserted_ood_k]


test_dirty_arr = test_dirty_arr[test_idx]
test_dirty_label = test_dirty_label[test_idx]
test_dirty_boundary = test_boundary_idx

ood_train_data = torchvision.datasets.FashionMNIST(
    './mnist', train=False, transform=torchvision.transforms.ToTensor()
)
ood_test_arr = np.asarray(ood_train_data.train_data.numpy())[ood_train_data.train_labels.numpy() == args.inserted_ood_k]
ood_test_label = np.asarray(ood_train_data.train_labels.numpy())[ood_train_data.train_labels.numpy() == args.inserted_ood_k]

test_dirty_boundary = test_dirty_boundary[test_dirty_label != args.inserted_ood_k]
test_dirty_arr = test_dirty_arr[test_dirty_label!=args.inserted_ood_k]
test_dirty_label = test_dirty_label[test_dirty_label != args.inserted_ood_k]

test_dirty_arr = np.vstack((test_dirty_arr, ood_test_arr))
test_dirty_label = np.hstack((test_dirty_label, ood_test_label))
test_dirty_boundary = np.hstack((test_dirty_boundary, [-1 for _ in range(len(ood_test_label))]))

test_arr = test_dirty_arr
test_label = test_dirty_label

np.save(f'data/DMNIST/{args.inserted_ood_k}_{args.validation_split}_test_arr', test_arr)
np.save(f'data/DMNIST/{args.inserted_ood_k}_{args.validation_split}_test_label', test_label)

np.save(f'data/DMNIST/{args.inserted_ood_k}_{args.validation_split}_valid_arr', valid_arr)
np.save(f'data/DMNIST/{args.inserted_ood_k}_{args.validation_split}_valid_label', valid_label)

np.save(f'data/DMNIST/{args.inserted_ood_k}_{args.validation_split}_train_arr', train_arr)
np.save(f'data/DMNIST/{args.inserted_ood_k}_{args.validation_split}_train_label', train_label)
np.save(f'data/DMNIST/{args.inserted_ood_k}_{args.validation_split}_test_dirty_boundary', test_dirty_boundary)

np.save(f'data/DMNIST/{args.inserted_ood_k}_{args.validation_split}_valid_dirty_boundary', valid_dirty_boundary)

import os
import sys
import torch
import torch.optim as optim
import logging
import numpy as np
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
from .eval import meta_test
sys.path.append('..')
from datasets import dataloaders


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def train_parser():
    parser = argparse.ArgumentParser()

    ## general hyper-parameters
    parser.add_argument("--opt", help="optimizer", choices=['adam','sgd'], default='sgd')
    parser.add_argument("--lr", help="initial learning rate", type=float, default=0.1)
    parser.add_argument("--gamma", help="learning rate cut scalar", type=float, default=0.1)
    parser.add_argument("--epoch", help="number of epochs before lr is cut by gamma", type=int, default=150)
    parser.add_argument("--weight_decay", help="weight decay for optimizer", type=float, default=5e-4)
    parser.add_argument('--gpu', default=2, type=int, help='gpu id')
    parser.add_argument("--seed", help="random seed", type=int, default=42)
    parser.add_argument("--val_epoch", help="number of epochs before eval on val", type=int, default=1)
    parser.add_argument("--resnet", help="whether use resnet12 as backbone or not", action="store_true")
    parser.add_argument("--nesterov", help="nesterov for sgd", action="store_true")
    parser.add_argument("--batch_size", help="batch size used during pre-training", type=int)
    parser.add_argument('--decay_epoch', nargs='+',help='epochs that cut lr', type=int)
    parser.add_argument("--pre", help="whether use pre-resized 84x84 images for val and test", action="store_true")
    parser.add_argument("--no_val", help="don't use validation set, just save model at final timestep", action="store_true")
    parser.add_argument("--train_way", help="training way", type=int)
    parser.add_argument("--test_way", help="test way", type=int, default=5)
    parser.add_argument("--train_shot", help="number of support images per class for meta-training and meta-testing during validation", type=int)
    parser.add_argument("--test_shot", nargs='+', help="number of support images per class for meta-testing during final test", type=int)
    parser.add_argument("--train_query_shot", help="number of query images per class during meta-training", type=int, default=15)
    parser.add_argument("--test_query_shot", help="number of query images per class during meta-testing", type=int, default=16)
    parser.add_argument("--train_transform_type", help="size transformation type during training", type=int)
    parser.add_argument("--test_transform_type", help="size transformation type during inference", type=int)
    parser.add_argument("--val_trial", help="number of meta-testing episodes during validation", type=int, default=1000)
    parser.add_argument("--detailed_name", help="whether include training details in the name", action="store_true")
    parser.add_argument("--dataset", choices=['stanford_dog', 'stanford_car', 'cub_raw', 'meta_iNat', 'tiered_meta_iNat'])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--model", choices=['C2_Net'], default='C2_Net')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_epoch", type=int, default=0)

    args = parser.parse_args()

    return args

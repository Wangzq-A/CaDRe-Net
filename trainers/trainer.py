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
    parser.add_argument("--opt",  choices=[])
    parser.add_argument("--lr",  type=float, default=0.1)
    parser.add_argument("--gamma",  type=float, default=0.1)
    parser.add_argument("--epoch",  type=int, default=150)
    parser.add_argument("--weight_decay",  type=float, default=5e-4)
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--val_epoch", default=1)
    parser.add_argument("--resnet",  action="store_true")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument('--decay_epoch', nargs='+', type=int)
    parser.add_argument("--pre",  action="store_true")
    parser.add_argument("--no_val",  action="store_true")
    parser.add_argument("--train_way", type=int)
    parser.add_argument("--test_way",  type=int, default=5)
    parser.add_argument("--train_shot", type=int)
    parser.add_argument("--test_shot", nargs='+',  type=int)
    parser.add_argument("--train_query_shot", type=int, default=15)
    parser.add_argument("--test_query_shot", type=int, default=16)
    parser.add_argument("--train_transform_type",  type=int)
    parser.add_argument("--test_transform_type",  type=int)
    parser.add_argument("--val_trial", type=int, default=1000)
    parser.add_argument("--detailed_name", action="store_true")
    parser.add_argument("--dataset", choices=['stanford_dog', 'stanford_car', 'cub_raw')
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--model")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_epoch", type=int, default=0)

    args = parser.parse_args()

    return args

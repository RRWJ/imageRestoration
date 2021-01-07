import imageio
import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
import os
from importlib import import_module
from option import args

if __name__ == '__main__':
    current_step = 0
    if args.phase == 'train':
        print('Load trainset...')
        from data.HRkerdataset import HRkerdataset as D
        datasets = D(args)
        print('Dataset [{:s} - {:s}] is created.'.format(datasets.__class__.__name__, args.dataset_mode))
        train_size = int(math.ceil(len(datasets) / args.batch_size))
        print('Number of train images: {:,d}, iters: {:,d}'.format(len(datasets), train_size))
        loader_train = DataLoader(datasets, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=not args.cpu, num_workers=args.n_threads, drop_last=True)
        # from model.tinynet import tinyNet
        # model = tinyNet(args) 传参报错
        # model = Model
        # print('Training model [{:s}] is created.'.format(model.__class__.__name__))
        # loss =
        for batch, train_data in enumerate(loader_train):
            current_step += 1
            print(train_data['LR'].shape)
            print(train_data['HR'].shape)

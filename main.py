import torch
import os
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from torch.utils.data import DataLoader

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)     ###setting the log and the train information

if checkpoint.ok:
    loader = data.Data(args)                ##data loader
    _model = model.Model(args, checkpoint)
    _loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, _model, _loss, checkpoint)
    while not t.terminate():
        # t.train()
        t.test()

    checkpoint.done()


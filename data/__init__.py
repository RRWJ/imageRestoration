from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torch
# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            print('Load trainset...')
            datasets = []
            for d in args.data_train:
                module_name = d
            # from data import renwenjia as module_train
            # + args.data_train.lower())
            # 1103dataloader如下写报错：
            # datasets.appends(getattr(module_train, args.data_train)(args))
                m = import_module('data.' + module_name.lower())
                datasets = getattr(m, module_name)(args,name=d, train=True)
            self.loader_train = DataLoader(
                datasets,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )
        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)
        self.loader_test.append(
            DataLoader(
                testset,
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )
        )

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()
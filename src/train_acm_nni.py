import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import collections
import torch
import numpy as np
import model.loss as module_loss
from data_loader.acm_dataset_x import ACMDataset
from config.parse_config import ConfigParser
from model.acm_model import MultiGraph
from torch_geometric.data import DataLoader
import random
from utils import prepare_device
from trainer.acm_nni_trainer import Trainer

import nni
from nni.utils import merge_parameter

# SEED = 123
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

def generate_default_params(args):
    '''
    Generate default hyper parameters
    '''
    return {
        'batch_size':args.batch_size,
        'learning_rate': args.learning_rate,
        'PRI_beta': 0.1,
        'PRI_alpha': 0.1,
        'PRI_weight': 0.05,
        'GIB_beta': 0.1,
        'GIB_cross_weight': 0.1
    }

def main(config,argx):
    try:
        tuner_params = nni.get_next_parameter()
        print("get_next_parameter")
        params = merge_parameter(generate_default_params(argx), tuner_params)

        logger = config.get_logger('train')

        data_set_train = ACMDataset(mode='train')
        data_set_train = DataLoader(data_set_train, batch_size=1, shuffle=True)
        data_set_val = ACMDataset(mode='test')
        data_set_val = DataLoader(data_set_val, batch_size=1, shuffle=True)
        data_set_test = ACMDataset(mode='test')
        data_set_test = DataLoader(data_set_test, batch_size=1, shuffle=True)

        model = MultiGraph()

        device, device_ids = prepare_device(config['n_gpu'])
        model = model.to(device)

        criterion = None #getattr(module_loss, config['loss'])
        metrics = None #[getattr(module_metric, met) for met in config['metrics']]

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(trainable_params,lr=params['learning_rate']) #config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = None #config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=config,
                          device=device,
                          train_data_set=data_set_train,
                          val_data_set=data_set_val,
                          test_data_set=data_set_test,
                          logger = logger,
                          params= params)
        trainer._test_epoch()
        trainer._valid_epoch(1)
        trainer.train()
        # trainer._test_epoch()
    except Exception as e:
        print(e)
        raise

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='../config/acm_config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    args.add_argument('--learning_rate', type=float, default=0.0008, metavar='LR',
                        help='learning rate (default: 0.01)')

    # custom cli options to modify configuration from default values given in json file.
    # CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    # options = [
    #     CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
    #     CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    # ]
    argx = args.parse_args()
    config = ConfigParser.from_args(args)
    main(config,argx)


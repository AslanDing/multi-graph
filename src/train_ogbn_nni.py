import argparse
import collections
import torch
import numpy as np
import model.loss as module_loss
from data_loader.ogbn_mag_dataset_x import OgbnMagDataset
from config.parse_config import ConfigParser
from model.ogbn_mag_model_softmax import MultiGraph
from torch_geometric.data import DataLoader
import random
from utils import prepare_device
from trainer.ogbn_trainer import Trainer

import nni
from nni.utils import merge_parameter

SEED = 12 #3
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
        'PRI_beta': 0.0,
        'PRI_alpha': 0.0,
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

        data_set_train = OgbnMagDataset(mode='train')
        data_set_train = DataLoader(data_set_train, batch_size=1, shuffle=True)
        data_set_val = OgbnMagDataset(mode='val')
        data_set_val = DataLoader(data_set_val, batch_size=1, shuffle=True)
        data_set_test = OgbnMagDataset(mode='test')
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
                          logger=logger,
                          params=params)
        trainer._valid_epoch(1)
        trainer.train()
    except Exception as e:
        print(e)
        raise
    # trainer._valid_epoch(1)
    # trainer._test_epoch()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='../config/ogbn_config.json', type=str, #
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=r'../exp/ogbn-mag/models/ogbn/1030_123850/checkpoint-epoch19.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    args.add_argument('--learning_rate', type=float, default=0.001, metavar='LR',
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


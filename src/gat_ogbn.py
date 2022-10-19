import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import InMemoryDataset, download_url
# from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv
from scipy.io import loadmat,savemat
from torch_geometric.data import Data,DataLoader,Dataset
from torch.nn import Linear, Parameter
from data_loader.ogbn_mag_dataset_x import OgbnMagDataset

import numpy as np
from sklearn.metrics import f1_score

# gdc =  True
lr = 1E-4  # 5E-3
epochs = 100

def label_to_vector(index):
    vec = np.zeros([1,4])
    vec[0,index]=1.0
    return vec


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_set_train = OgbnMagDataset(mode='train')
data_set_train = DataLoader(data_set_train, batch_size=1, shuffle=True)

data_set_test = OgbnMagDataset(mode='test')
data_set_test = DataLoader(data_set_test, batch_size=1, shuffle=True)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        #self.node_encoder = Linear(in_channels, hidden_channels)
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        #x = self.node_encoder(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x.softmax(dim=1)

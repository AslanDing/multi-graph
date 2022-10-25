import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import InMemoryDataset, download_url
# from torch_geometric.logging import init_wandb, log
from model.han_layer import HANConv
from scipy.io import loadmat,savemat
from torch_geometric.data import HeteroData,DataLoader,Dataset
from data_loader.ogbn_mag_dataset_x import OgbnMagDataset

import numpy as np
from sklearn.metrics import f1_score

# gdc =  True
lr = 5E-3 # 1E-4  #
epochs = 100

def label_to_vector(index,len=5):
    vec = np.zeros([1,len])
    vec[0,index]=1.0
    return vec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_set_train = OgbnMagDataset(mode='train')
data_set_train = DataLoader(data_set_train, batch_size=1, shuffle=True)

data_set_test = OgbnMagDataset(mode='val')
data_set_test = DataLoader(data_set_test, batch_size=1, shuffle=True)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        metadata =(["paper","author"],
                   [('paper', 'subject', 'paper'),
                    ('author', 'Institute', 'author'),
                    ('author', 'write', 'paper'),
                    ('paper', 'from', 'author')])
        self.conv1 = HANConv(in_channels, hidden_channels,metadata, heads, dropout=0.5)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = HANConv(hidden_channels , out_channels,metadata, heads=1, dropout=0.5)

    def forward(self, x, edge_index):
        for key in x.keys():
            x[key] = F.dropout(x[key], p=0.5, training=self.training)
        x =self.conv1(x, edge_index)
        for key in x.keys():
            x[key] = F.elu(x[key])
            x[key] = F.dropout(x[key], p=0.5, training=self.training)
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GAT(128, 256, 5,4)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
batch_size = 32
length = len(data_set_train)

def train():
    iter_data = iter(data_set_train)
    total_loss = 0
    for batch_idx in range(length // batch_size):
        optimizer.zero_grad()
        for i in range(batch_size):
            data = next(iter_data)
            data = data.to(device)
            x_dict = data.x_dict
            edge_dict = data.edge_index_dict
            batch_dict = data.batch_dict
            y_dict = data.y_dict
            mask_dict = data.mask_dict

            out = model(x_dict, edge_dict)

            loss = F.cross_entropy(out["paper"][mask_dict["paper"]], y_dict["paper"])
            # optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1,norm_type=2)
            # optimizer.step()
            total_loss += loss.item()
            # print(loss.item(),end=" ")
        optimizer.step()
        # print()

    return float(total_loss/length)

@torch.no_grad()
def test():
    model.eval()
    y_hats = []
    y_labels = []
    acc = []

    iter_data = iter(data_set_test)
    for batch_idx, data in enumerate(data_set_test):
        data = data.to(device)
        x_dict = data.x_dict
        edge_dict = data.edge_index_dict
        batch_dict = data.batch_dict
        y_dict = data.y_dict
        mask_dict = data.mask_dict

        out = model(x_dict, edge_dict)
        out = out["paper"][mask_dict["paper"]]
        acc.append((torch.argmax(out, 1) == torch.argmax(data.y_dict['paper'], 1)).cpu().detach().numpy())
        y_hats.append(torch.argmax(out, 1).cpu().detach().numpy())
        y_labels.append(torch.argmax(data.y_dict['paper'], 1).cpu().detach().numpy())

    #acc = np.array(acc).mean()
    micre_f1 = f1_score(np.array(y_labels), np.array(y_hats), average="micro")
    macre_f1 = f1_score(np.array(y_labels), np.array(y_hats), average="macro")

    return micre_f1,macre_f1

bmic_list = []
bmac_list = []
for i in range(10):
    bmic = 0
    bmac = 0
    for epoch in range(1, epochs + 1):
        loss = train()
        micre_f1,macre_f1 = test()
        if micre_f1 > bmic:
            bmic = micre_f1
        if macre_f1 > bmac:
            bmac = macre_f1
        print(
            f'Epoch: {epoch:02d}, Loss: {loss:.4f}, bmic:{bmic:.4f}, bmac:{bmac:.4f}')
    bmic_list.append(bmic)
    bmac_list.append(bmac)

print(bmic_list)
print(np.array(bmic_list).mean())
print(bmac_list)
print(np.array(bmac_list).mean())

"""
bmic:0.1329, bmac:0.0558
"""

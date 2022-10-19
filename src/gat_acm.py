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
from data_loader.acm_dataset_x import ACMDataset

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

data_set_train = ACMDataset(mode='train')
data_set_train = DataLoader(data_set_train, batch_size=1, shuffle=True)

data_set_test = ACMDataset(mode='test')
data_set_test = DataLoader(data_set_test, batch_size=1, shuffle=True)

# data_dir = r'/media/aslan/50E4BE16E4BDFDF2/DATA/CODE/new_project/datasets/acm'
# data_path = data_dir + r'/ACM_multi_graph.mat'
#
# datas = loadmat(data_path)
# # feature0 = datas['NF0']
# feature1 = datas['PT']
# # feature2 = datas['NF2']
# # feature3 = datas['NF3']
#
# # A00 = datas['A00']
# A11 = datas['PSP']
# A11_index = A11.nonzero()
# # A22 = datas['A22']
# # A33 = datas['A33']
#
# train_list = datas['train_idx']
# train_label = datas['train_taget']  # not one hot
# train_label_oh  = np.zeros((train_label.shape[1],4))
# for i in range(train_label.shape[1]):
#     train_label_oh[i] = label_to_vector(train_label[:,i])
# train_label_th = torch.from_numpy(train_label_oh).to(device)
# test_list = datas['test_idx']
# test_label = datas['test_taget']
# test_label_oh  = np.zeros((test_label.shape[1],4))
# for i in range(test_label.shape[1]):
#     test_label_oh[i] = label_to_vector(test_label[:,i])
# test_label_th = torch.from_numpy(test_label_oh).to(device)
# data = Data(x = torch.from_numpy(feature1).float(),
#             edge_index=torch.stack([torch.from_numpy(A11_index[0].astype(np.int64)),
#                                     torch.from_numpy(A11_index[1].astype(np.int64))], dim=0))



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

model = GAT(1902, 256, 3,3)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
batch_size = 32
length = len(data_set_train)

# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index)
#     loss = F.cross_entropy(out[train_list], train_label_th)
#     loss.backward()
#     optimizer.step()
#     return float(loss)

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

            out = model(x_dict["paper"], edge_dict[("paper","subject","paper")])
            loss = F.cross_entropy(out[mask_dict["paper"]], y_dict["paper"])
            loss.backward()
            total_loss += loss.item()
        optimizer.step()

    return float(total_loss)

# @torch.no_grad()
# def test():
#     model.eval()
#     pred = model(data.x, data.edge_index).argmax(dim=-1)
#
#     micre_f1 = f1_score(pred[test_list].cpu().detach().numpy(), test_label[0], average="micro")
#     macre_f1 = f1_score(pred[test_list].cpu().detach().numpy(), test_label[0], average="macro")
#     # accs = []
#     # for mask in [data.train_mask, data.val_mask, data.test_mask]:
#     #     accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
#     return micre_f1,macre_f1

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

        out = model(x_dict["paper"], edge_dict[("paper", "subject", "paper")])
        out = out[mask_dict["paper"]]
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
 bmic:0.6637, bmac:0.6684

"""

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
from data_loader.pubmed_dataset_x import PubMedDataset
from torch.nn import Linear, Parameter

import numpy as np
from sklearn.metrics import f1_score

# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randn(3, 5).softmax(dim=1)
# loss = F.cross_entropy(input, target)
# loss.backward()

# gdc =  True
lr = 5E-3 # 1E-4  #
epochs = 100

def label_to_vector(index):
    vec = np.zeros([1,8])
    vec[0,index]=1.0
    return vec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_set_train = PubMedDataset(mode='train')
data_set_train = DataLoader(data_set_train, batch_size=1, shuffle=True)

data_set_test = PubMedDataset(mode='test')
data_set_test = DataLoader(data_set_test, batch_size=1, shuffle=True)

# data_dir = r'/media/aslan/50E4BE16E4BDFDF2/DATA/CODE/HNE-master/Model/MAGNN/data/PubMed'
# data_path = data_dir + r'/process.mat'
#
# datas = loadmat(data_path)
# feature0 = datas['NF0']
# feature1 = datas['NF1']
# feature2 = datas['NF2']
# feature3 = datas['NF3']

# A00 = datas['A00']
# A00_index = A00.nonzero()
# A11 = datas['A11']
# A11_index = A11.nonzero()
# A22 = datas['A22']
# A22_index = A22.nonzero()
# A33 = datas['A33']
# A33_index = A33.nonzero()
#
# A01 = datas['A01']
# A01_index = A01.nonzero()
# # A20 = datas['A20']
# # A30 = datas['A30']
# A21 = datas['A21']
# A21_index = A21.nonzero()
# A31 = datas['A31']
# A31_index = A31.nonzero()
# A23 = datas['A23']
# A23_index = A23.nonzero()

# data = HeteroData()
# data['domain0'].x = torch.from_numpy(feature0).float()
# data['domain1'].x = torch.from_numpy(feature1).float()
# data['domain2'].x = torch.from_numpy(feature2).float()
# data['domain3'].x = torch.from_numpy(feature3).float()
# data['domain0', 'to', 'domain0'].edge_index = torch.stack([torch.from_numpy(A00_index[0].astype(np.int64)),
#                                                                    torch.from_numpy(A00_index[1].astype(np.int64))], dim=0)
# data['domain1', 'to', 'domain1'].edge_index = torch.stack([torch.from_numpy(A11_index[0].astype(np.int64)),
#                                                                    torch.from_numpy(A11_index[1].astype(np.int64))], dim=0)
# data['domain2', 'to', 'domain2'].edge_index = torch.stack([torch.from_numpy(A22_index[0].astype(np.int64)),
#                                                                    torch.from_numpy(A22_index[1].astype(np.int64))], dim=0)
# data['domain3', 'to', 'domain3'].edge_index = torch.stack([torch.from_numpy(A33_index[0].astype(np.int64)),
#                                                                    torch.from_numpy(A33_index[1].astype(np.int64))], dim=0)
#
# data['domain0', 'to', 'domain1'].edge_index = torch.stack([torch.from_numpy(A01_index[0].astype(np.int64)),
#                                                                    torch.from_numpy(A01_index[1].astype(np.int64))], dim=0)
# data['domain1', 'to', 'domain0'].edge_index = torch.stack([torch.from_numpy(A01_index[1].astype(np.int64)),
#                                                                    torch.from_numpy(A01_index[0].astype(np.int64))], dim=0)
#
# data['domain2', 'to', 'domain1'].edge_index = torch.stack([torch.from_numpy(A21_index[0].astype(np.int64)),
#                                                                    torch.from_numpy(A21_index[1].astype(np.int64))], dim=0)
# data['domain1', 'to', 'domain2'].edge_index = torch.stack([torch.from_numpy(A21_index[1].astype(np.int64)),
#                                                                    torch.from_numpy(A21_index[0].astype(np.int64))], dim=0)
# data['domain3', 'to', 'domain1'].edge_index = torch.stack([torch.from_numpy(A31_index[0].astype(np.int64)),
#                                                                    torch.from_numpy(A31_index[1].astype(np.int64))], dim=0)
# data['domain1', 'to', 'domain3'].edge_index = torch.stack([torch.from_numpy(A31_index[1].astype(np.int64)),
#                                                                    torch.from_numpy(A31_index[0].astype(np.int64))], dim=0)
#
# train_list = datas['train_list']
# train_label = datas['train_label']  # not one hot
# train_label_oh  = np.zeros((train_label.shape[1],8))
# for i in range(train_label.shape[1]):
#     train_label_oh[i] = label_to_vector(train_label[:,i])
# train_label_th = torch.from_numpy(train_label_oh).to(device)
# test_list = datas['test_list']
# test_label = datas['test_label']
# test_label_oh  = np.zeros((test_label.shape[1],8))
# for i in range(test_label.shape[1]):
#     test_label_oh[i] = label_to_vector(test_label[:,i])
# test_label_th = torch.from_numpy(test_label_oh).to(device)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        metadata =(["domain0","domain1","domain2","domain3"],
                   [('domain0', 'to', 'domain0'),
                    ('domain1', 'to', 'domain1'),
                    ('domain2', 'to', 'domain2'),
                    ('domain3', 'to', 'domain3'),

                    ('domain0', 'to', 'domain1'),
                    ('domain1', 'to', 'domain0'),
                    ('domain2', 'to', 'domain1'),
                    ('domain1', 'to', 'domain2'),
                    ('domain3', 'to', 'domain1'),
                    ('domain1', 'to', 'domain3')])
        self.conv1 = HANConv(in_channels, hidden_channels,metadata, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = HANConv(hidden_channels , out_channels,metadata, heads=1, dropout=0.6)
        # self.linear = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for key in x.keys():
            x[key] = F.dropout(x[key], p=0.6, training=self.training)
        x =self.conv1(x, edge_index)
        for key in x.keys():
            if x[key] == None:
                continue
            x[key] = F.elu(x[key])
            x[key] = F.dropout(x[key], p=0.6, training=self.training)
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        for key in x.keys():
            if x[key] == None:
                continue
            x[key] = x[key].softmax(dim=1)
        return x

model = GAT(200, 256, 8,8)
model = model.to(device)

def print_grad(grad):
    print(grad.max(),grad.min(),grad.mean(),grad.shape)

hooks = {}
# for idx , item in enumerate(model.named_parameters()):
#     if idx == 0:
#         print(item[0])
#         hooks[item[0]] = item[1].register_hook(print_grad)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
batch_size = 32
length = len(data_set_train)

# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x_dict, data.edge_index_dict)
#     loss = F.cross_entropy(out['domain1'][train_list], train_label_th)
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

            out = model(x_dict, edge_dict)
            loss = F.cross_entropy(out["domain1"][mask_dict["domain1"]], y_dict["domain1"])
            # optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1,norm_type=2)
            # optimizer.step()
            total_loss += loss.item()
            # print(loss.item(),end=" ")
        optimizer.step()
        # print()

    return float(total_loss)

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
        out = out["domain1"][mask_dict["domain1"]]
        acc.append((torch.argmax(out, 1) == torch.argmax(data.y_dict['domain1'], 1)).cpu().detach().numpy())
        y_hats.append(torch.argmax(out, 1).cpu().detach().numpy())
        y_labels.append(torch.argmax(data.y_dict['domain1'], 1).cpu().detach().numpy())

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
bmic:0.5233, bmac:0.3948
"""

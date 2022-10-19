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
from data_loader.pubmed_dataset_x import PubMedDataset

import numpy as np
from sklearn.metrics import f1_score

# gdc =  True
lr = 1E-4  # 5E-3
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
# # feature0 = datas['NF0']
# feature1 = datas['NF1']
# # feature2 = datas['NF2']
# # feature3 = datas['NF3']
#
# # A00 = datas['A00']
# A11 = datas['A11']
# A11_index = A11.nonzero()
# # A22 = datas['A22']
# # A33 = datas['A33']
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
# data = Data(x = torch.from_numpy(feature1).float(),
#             edge_index=torch.stack([torch.from_numpy(A11_index[0].astype(np.int64)),
#                                     torch.from_numpy(A11_index[1].astype(np.int64))], dim=0))


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x.softmax(dim=1)

model = GAT(200, 256, 8,8)
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

            out = model(x_dict["domain1"], edge_dict[("domain1","to","domain1")])
            loss = F.cross_entropy(out[mask_dict["domain1"]], y_dict["domain1"])
            loss.backward()
            total_loss += loss.item()
        optimizer.step()

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

        out = model(x_dict["domain1"], edge_dict[("domain1", "to", "domain1")])
        out = out[mask_dict["domain1"]]
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
[0.5581395348837209, 0.5232558139534884, 0.47674418604651164, 0.5348837209302325, 0.5116279069767442, 0.5116279069767442, 0.5116279069767442, 0.5232558139534884, 0.5581395348837209, 0.5232558139534884]
0.5232558139534884
[0.5191680602006689, 0.4699442327538303, 0.4273695054945055, 0.47570436507936503, 0.46062329483382114, 0.4665716436696296, 0.46973728287168137, 0.4756798756798757, 0.5074820062977957, 0.491812097646076]
0.4764092364527249

[0.3720930232558139, 0.40697674418604657, 0.4418604651162791, 0.46511627906976744, 0.5, 0.5116279069767442, 0.4883720930232558, 0.5, 0.5, 0.5]
0.4686046511627907
[0.2706547619047619, 0.34723193473193475, 0.39685891431324954, 0.40750740487582593, 0.4305776198439242, 0.43938862501522347, 0.4273753720644705, 0.42709186948317385, 0.42709186948317385, 0.42473262032085557]
0.39985109920365935

bmic:0.4884, bmac:0.4210
bmic:0.5116, bmac:0.4561
bmic:0.5116, bmac:0.4629
bmic:0.5000, bmac:0.4379
"""

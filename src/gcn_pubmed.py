import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import InMemoryDataset, download_url
# from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
from scipy.io import loadmat,savemat
from torch_geometric.data import Data,DataLoader,Dataset
from data_loader.pubmed_dataset_x import PubMedDataset

import numpy as np
from sklearn.metrics import f1_score

gdc =  False
lr = 5E-4  # 1E-4 #
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
#
#
# if gdc:
#     transform = T.GDC(
#         self_loop_weight=1,
#         normalization_in='sym',
#         normalization_out='col',
#         diffusion_kwargs=dict(method='ppr', alpha=0.05),
#         sparsification_kwargs=dict(method='topk', k=128, dim=0),
#         exact=True,
#     )
#     data = transform(data)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True,
                             normalize=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True,
                             normalize=False)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x.softmax(dim=-1)

model = GCN(200, 256, 8)
model = model.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=lr)  # Only perform weight-decay on first convolution.
batch_size = 368
length = len(data_set_train)

# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index, data.edge_weight)
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

            out = model(x_dict["domain1"], edge_dict[("domain1","to","domain1")])
            loss = F.cross_entropy(out[mask_dict["domain1"]], y_dict["domain1"])
            loss.backward()
            total_loss += loss.item()
        # total_loss /= batch_size
        # total_loss.backward()
        optimizer.step()

    return float(total_loss)

# @torch.no_grad()
# def test():
#     model.eval()
#     pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=-1)
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
[0.20930232558139536, 0.22093023255813954, 0.19767441860465115, 0.19767441860465115, 0.20930232558139536, 0.20930232558139536, 0.20930232558139536, 0.20930232558139536, 0.20930232558139536, 0.20930232558139536]
0.2081395348837209
[0.10621639784946237, 0.07107843137254902, 0.104538980874473, 0.1059375, 0.10095817963335568, 0.12041666666666667, 0.10238516227605411, 0.1198809523809524, 0.10862337294850029, 0.10138146167557932]
0.1041417105677593

[0.22093023255813954, 0.22093023255813954, 0.22093023255813954, 0.22093023255813954, 0.22093023255813954, 0.22093023255813954, 0.22093023255813954, 0.22093023255813954, 0.22093023255813954, 0.22093023255813954]
0.22093023255813954
[0.08750000000000001, 0.045238095238095244, 0.045238095238095244, 0.045238095238095244, 0.045238095238095244, 0.045238095238095244, 0.045238095238095244, 0.045238095238095244, 0.045238095238095244, 0.045238095238095244]
0.049464285714285725

[0.36046511627906974, 0.4186046511627907, 0.43023255813953487, 0.46511627906976744, 0.46511627906976744, 0.46511627906976744, 0.46511627906976744, 0.46511627906976744, 0.46511627906976744, 0.46511627906976744]
0.44651162790697674
[0.2798455091550621, 0.3669525826973321, 0.38827641285268405, 0.42901922638764745, 0.42901922638764745, 0.4397986321524552, 0.4397986321524552, 0.4257500563952177, 0.4232638430751638, 0.42184338852970926]
0.4043567509785374

[0.5465116279069767, 0.5465116279069767, 0.5465116279069767, 0.5813953488372093, 0.5581395348837209, 0.5465116279069767, 0.5348837209302325, 0.5465116279069767, 0.5465116279069767, 0.5465116279069767]
0.55
[0.5099662137062022, 0.47519721758852196, 0.5018971224853578, 0.5236371815319183, 0.502569498579224, 0.49544293425872377, 0.47389607343875634, 0.4856566622016334, 0.49445951900220186, 0.490419658666142]
0.49531420814586813

bmic:0.3837, bmac:0.3495

"""

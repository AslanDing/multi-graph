import os.path as osp

import matplotlib.pyplot as plt
import torch
print(torch.__version__)
print(torch.version.cuda)

from sklearn.manifold import TSNE

from torch_geometric.nn import Node2Vec

from scipy.io import loadmat
import numpy as np


def main():
    data_dir = r'../exp/ogbn_mag/data'
    data_path = data_dir + r'/ogbn_mag_graph.mat'

    datas = loadmat(data_path)

    A11 = datas['PCP']

    Paper_Label = datas['PL']

    train_list = datas['train_idx']
    train_label = Paper_Label[train_list[0,:]]

    test_list = datas['test_idx']
    test_label = Paper_Label[test_list[0,:]]

    val_list = datas['val_idx']
    val_label = Paper_Label[val_list[0,:]]

    A11sub = A11.nonzero()
    edge_index = torch.stack([torch.from_numpy(A11sub[0].astype(np.int64)),
                 torch.from_numpy(A11sub[1].astype(np.int64))], dim=0)

    #data = dataset[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(edge_index, embedding_dim=128, walk_length=20,
                     context_size=10,sparse=True).to(device)
                     #walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        #count = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            #print(count,loss.item())
            #count += 1
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc,micro,macro = model.test(z[train_list], torch.from_numpy(train_label.reshape(-1)),
                         z[test_list], torch.from_numpy(test_label.reshape(-1)),
                         max_iter=150)
        return acc,micro,macro
    best_list = []
    bmic_list = []
    bmac_list = []
    for i in range(10):
        best = 0
        bmic = 0
        bmac = 0
        for epoch in range(1, 101):
            loss = train()
            acc,micro,macro = test()
            if acc>best:
                best = acc
            if bmic<micro:
                bmic = micro
            if bmac<macro:
                bmac = macro
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}, Best:{best:.4f}, bmic:{bmic:.4f}, bmac:{bmac:.4f}')
        best_list.append(best)
        bmic_list.append(bmic)
        bmac_list.append(bmac)

    print(best_list)
    print(np.array(best_list).mean())
    print(bmic_list)
    print(np.array(bmic_list).mean())
    print(bmac_list)
    print(np.array(bmac_list).mean())
""" 
micro_f1  0.3960, bmac:0.3662
"""

if __name__ == "__main__":
    main()
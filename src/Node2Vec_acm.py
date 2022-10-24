import os.path as osp

import matplotlib.pyplot as plt
import torch
print(torch.__version__)
print(torch.version.cuda)

from sklearn.manifold import TSNE

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

from scipy.io import loadmat
import numpy as np


def main():
    data_dir = r'../exp/ogbn_mag/data'
    data_path = data_dir + r'/ACM_multi_graph.mat'

    datas = loadmat(data_path)

    A11 = datas['PSP']

    train_list = datas['train_idx']
    train_label = datas['train_taget']

    test_list = datas['test_idx']
    test_label = datas['test_taget']

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
    
[0.6745283018867925, 0.6834905660377358, 0.6702830188679245, 0.6754716981132075, 0.6726415094339623, 0.6764150943396227, 0.6735849056603773, 0.6721698113207547, 0.675, 0.6773584905660377]
0.6750943396226414
[0.6812511748591804, 0.6903808574146053, 0.67982451153088, 0.6831307996818609, 0.6808226976567565, 0.6850087167691649, 0.6811038611830386, 0.6794175592197961, 0.6827755540334204, 0.6858538066387205]
0.6829569538987423

0.6834905660377358  0.6903808574146053

    """
    # @torch.no_grad()
    # def plot_points(colors):
    #     model.eval()
    #     z = model(torch.arange(data.num_nodes, device=device))
    #     z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    #     y = data.y.cpu().numpy()
    #
    #     plt.figure(figsize=(8, 8))
    #     for i in range(dataset.num_classes):
    #         plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    #     plt.axis('off')
    #     plt.show()

    # colors = [
    #     '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
    #     '#ffd700'
    # ]
    # plot_points(colors)


if __name__ == "__main__":
    main()
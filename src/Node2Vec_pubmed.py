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
    data_dir = r'/media/aslan/50E4BE16E4BDFDF2/DATA/CODE/HNE-master/Model/MAGNN/data/PubMed'
    data_path = data_dir + r'/process.mat'
    datas = loadmat(data_path)

    A11 = datas['A11']

    train_list = datas['train_list']
    train_label = datas['train_label']

    test_list = datas['test_list']
    test_label = datas['test_label']

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
    [0.23255813953488372, 0.26744186046511625, 0.22093023255813954, 0.22093023255813954, 0.20930232558139536, 0.20930232558139536, 0.22093023255813954, 0.19767441860465115, 0.20930232558139536, 0.23255813953488372]
    0.22209302325581395
    [0.23255813953488372, 0.26744186046511625, 0.22093023255813954, 0.22093023255813954, 0.20930232558139536, 0.20930232558139536, 0.22093023255813954, 0.19767441860465115, 0.20930232558139536, 0.23255813953488372]
    0.22209302325581395
    [0.2081439393939394, 0.22324862764734243, 0.18409600502706014, 0.17393557422969186, 0.16610402268297006, 0.19576176384687022, 0.1849296536796537, 0.16908040969367627, 0.19579162667397962, 0.18227850778254004]
    0.18833701306577239
    
    [0.26744186046511625, 0.2441860465116279, 0.22093023255813954, 0.20930232558139536, 0.23255813953488372, 0.22093023255813954, 0.23255813953488372, 0.23255813953488372, 0.27906976744186046, 0.23255813953488372]
    0.2372093023255814
    [0.26744186046511625, 0.2441860465116279, 0.22093023255813954, 0.20930232558139536, 0.23255813953488372, 0.22093023255813954, 0.23255813953488372, 0.23255813953488372, 0.27906976744186046, 0.23255813953488372]
    0.2372093023255814
    [0.22657542322176466, 0.1889017392632521, 0.17293147109241078, 0.19691738355421218, 0.19790490555196438, 0.1749461482776277, 0.20126590463799768, 0.18531344523570933, 0.20443166582872466, 0.21074570853982622]
    0.19599337952034895
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
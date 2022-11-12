import networkx
import numpy as np

from ge.classify import read_node_label, Classifier
from ge import DeepWalk
from sklearn.linear_model import LogisticRegression

from scipy.io import loadmat

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

def main():
    data_dir = r'../exp/ogbn_mag/data'
    data_path = data_dir + r'/ogbn_mag_graph.mat'

    datas = loadmat(data_path)
    A11 = datas['PCP']

    Paper_Label = datas['PL']
    train_list = datas['train_idx'][0].tolist()
    train_label = Paper_Label[train_list][:,0].tolist()

    val_list = datas['test_idx'][0].tolist()
    val_label = Paper_Label[val_list][:,0].tolist()

    test_list = datas['val_idx'][0].tolist()
    test_label = Paper_Label[test_list][:,0].tolist()

    G = networkx.Graph()

    node_nums = A11.shape[0]
    for i in range(node_nums):
        G.add_node(str(i))
    A11sub = A11.nonzero()
    for i in range(len(A11sub[0])):
        G.add_edge(str(A11sub[0][i]), str(A11sub[1][i]))

    model = DeepWalk(G, walk_length=20, num_walks=80, workers=1)
    model.train(window_size=5, iter=3)

    embeddings = model.get_embeddings()
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())

    Y_all = [[i] for i in train_label]
    Y_all.extend([[i] for i in test_label])
    train_label_list = [[i] for i in train_label]
    test_label_list = [[i] for i in test_label]

    clf.train(train_list, train_label_list,Y_all)

    clf.evaluate(test_list,test_label_list)

if __name__=="__main__":
    main()
"""
{'micro': 0.5357815442561206, 'macro': 0.4900085033178752, 'samples': 0.5357815442561206, 'weighted': 0.5674783247527964, 'acc': 0.5357815442561206}

{'micro': 0.5249861188228762, 'macro': 0.5155444522244479, 'samples': 0.5249861188228762, 'weighted': 0.5355726843098352, 'acc': 0.5249861188228762}

"""


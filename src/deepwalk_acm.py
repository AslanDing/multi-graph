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
    data_dir = r'../exp/acm/data'
    data_path = data_dir + r'/ACM_multi_graph.mat'

    datas = loadmat(data_path)
    A11 = datas['PSP']

    train_list = datas['train_idx'][0].tolist()
    train_label = datas['train_taget'][0].tolist()

    test_list = datas['test_idx'][0].tolist()
    test_label = datas['test_taget'][0].tolist()

    G = networkx.Graph()

    node_nums = A11.shape[0]
    for i in range(node_nums):
        G.add_node(str(i))
    A11sub = A11.nonzero()
    for i in range(len(A11sub[0])):
        G.add_edge(str(A11sub[0][i]), str(A11sub[1][i]))

    model = DeepWalk(G, walk_length=20, num_walks=8, workers=1)
    model.train(window_size=10, iter=3)

    embeddings = model.get_embeddings()
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())


    Y_all = [ [i] for i in train_label]
    Y_all.extend([ [i] for i in test_label])
    train_label_list = [ [i] for i in train_label]
    test_label_list = [ [i] for i in test_label]

    clf.train(train_list, train_label_list,Y_all)

    clf.evaluate(test_list,test_label_list)

if __name__=="__main__":
    main()
    """
    {'micro': 0.6721698113207547, 'macro': 0.668548672485029, 'samples': 0.6721698113207547, 'weighted': 0.6585603982219734, 'acc': 0.6721698113207547}
    """


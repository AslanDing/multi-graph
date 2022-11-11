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
    data_dir = r'../exp/pubmed/data'
    data_path = data_dir + r'/PubMed_process.mat'

    datas = loadmat(data_path)
    A11 = datas['A11']

    train_list = datas['train_list'][0].tolist()
    train_label = datas['train_label'][0].tolist()

    test_list = datas['test_list'][0].tolist()
    test_label = datas['test_label'][0].tolist()

    node_nums = A11.shape[0]
    A11sub = A11.nonzero()
    G = networkx.Graph()
    for i in range(len(A11sub[0])):
        G.add_edge(str(A11sub[0][i]),str(A11sub[1][i]))

    model = DeepWalk(G, walk_length=20, num_walks=8, workers=1)
    model.train(window_size=10, iter=3)

    embeddings = model.get_embeddings()
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.train(train_list, train_label)

    clf.evaluate(test_list,test_label)

if __name__=="__main__":
    main()



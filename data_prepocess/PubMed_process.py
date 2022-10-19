import numpy as np
import torch
from scipy.io import loadmat, savemat

from scipy.sparse import csr_matrix, coo_matrix

from collections import defaultdict

"""
TYPE	MEANING
0		GENE
1		DISEASE
2		CHEMICAL
3		SPECIES

LINK	START	END	MEANING
0		0		0	GENE-and-GENE
1		0		1	GENE-causing-DISEASE
2		1		1	DISEASE-and-DISEASE
3		2		0	CHEMICAL-in-GENE
4		2		1	CHEMICAL-in-DISEASE
5		2		2	CHEMICAL-and-CHEMICAL
6		2		3	CHEMICAL-in-SPECIES
7		3		0	SPECIES-with-GENE
8		3		1	SPECIES-with-DISEASE
9		3		3	SPECIES-and-SPECIES

TYPE	CLASS	MEANING
1		0		cardiovascular_disease
1		1		glandular_disease
1		2		nervous_disorder
1		3		communicable_disease
1		4		inflammatory_disease
1		5		pycnosis
1		6		skin_disease
1		7		cancer
"""


def read_data_(nodefile, linkfile, pathfile, labelfile, attributed, supervised):
    ntype_set = [[], [], [], []]
    ntype_dict = [{}, {}, {}, {}]
    node_features = [[], [], [], []]
    with open(nodefile, 'r') as file:
        for line in file:
            line = line[:-1].split('\t')
            node, ntype = int(line[0]), int(line[1])

            ntype_set[ntype].append(node)

            if attributed == 'True':
                node_features[ntype].append(np.array(line[2].split(',')).astype(np.float32))
    for i in range(4):
        for index, j in enumerate(ntype_set[i]):
            ntype_dict[i][j] = index

    # del ntype_set

    link_list = [
        [[], []],
        [[], []],
        [[], []],
        [[], []],
        [[], []],
        [[], []],
        [[], []],
        [[], []],
        [[], []],
        [[], []]]
    link_type = [[0, 0],
                 [0, 1],
                 [1, 1],
                 [2, 0],
                 [2, 1],
                 [2, 2],
                 [2, 3],
                 [3, 0],
                 [3, 1],
                 [3, 3]]
    with open(linkfile, 'r') as file:
        for line in file:
            left, right, ltype = list(map(int, line[:-1].split('\t')))
            dictmap_left = ntype_dict[link_type[ltype][0]]
            dictmap_right = ntype_dict[link_type[ltype][1]]
            try:
                link_list[ltype][0].append(dictmap_left[left])
                link_list[ltype][1].append(dictmap_right[right])
            except Exception:
                print(ltype)
                pass

    link_matrix = []
    for i in range(len(link_list)):
        adj_coo = link_list[i]
        row = np.array(adj_coo[0])
        col = np.array(adj_coo[1])
        data = np.ones_like(row)

        dim0 = len(ntype_set[link_type[i][0]])
        dim1 = len(ntype_set[link_type[i][1]])

        csr_matrix = coo_matrix((data, (row, col)), shape=(dim0, dim1)).tocsr()
        link_matrix.append(csr_matrix)
    del link_list
    label_train_list = []
    node_train_labels = []
    with open(labelfile, 'r') as file:
        for line in file:
            node, label = line[:-1].split('\t')
            node = ntype_dict[1][int(node)]
            node_train_labels.append(int(label))
            label_train_list.append(node)
    label_train_list = np.array(label_train_list)
    node_train_labels = np.array(node_train_labels)

    label_test_list = []
    node_test_labels = []
    with open(r'/media/aslan/50E4BE16E4BDFDF2/DATA/DATASET/PubMed/label.dat.test', 'r') as file:
        for line in file:
            node, _, type, label = line[:-1].split('\t')
            node = ntype_dict[1][int(node)]
            node_test_labels.append(int(label))
            label_test_list.append(node)

    label_test_list = np.array(label_test_list)
    node_test_labels = np.array(node_test_labels)

    savemat('../exp/pubmed/data/PubMed_process.mat',
            {'NF0': node_features[0],
             'NF1': node_features[1],
             'NF2': node_features[2],
             'NF3': node_features[3],

             'A00': link_matrix[0],
             'A11': link_matrix[2],
             'A22': link_matrix[5],
             'A33': link_matrix[9],

             'A01': link_matrix[1],
             'A20': link_matrix[3],
             'A30': link_matrix[7],

             'A21': link_matrix[4],
             'A31': link_matrix[8],

             'A23': link_matrix[6],

             'train_list': label_train_list,
             'train_label': node_train_labels,

             'test_list': label_test_list,
             'test_label': node_test_labels
             })


def preprocess(dir):
    node = dir + '/node.dat'
    link = dir + '/link.dat'
    path = dir + '/path.dat'
    label = dir + '/label.dat'
    read_data_(nodefile=node, linkfile=link, pathfile=path, labelfile=label, attributed='True', supervised='True')


if __name__ == "__main__":
    dir = r'/media/aslan/50E4BE16E4BDFDF2/DATA/CODE/HNE-master/Model/MAGNN/data/PubMed'
    preprocess(dir)
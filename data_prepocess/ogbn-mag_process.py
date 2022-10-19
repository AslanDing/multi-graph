import numpy as np
import torch
from scipy.io import loadmat,savemat

from scipy.sparse import csr_matrix,coo_matrix

import gzip
import random

SEED = 12
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

def read_gz(file):
    lines = []
    with gzip.open(file) as f:
        for line in f:
            line=line[:-1]
            lines.append(int(line))
    return lines

def propcess(path= './ogbn_mag/processed/geometric_data_processed.pt',
             tain_file = r'/media/aslan/50E4BE16E4BDFDF2/DATA/CODE/new_project/datasets/ogbn-mag/ogbn_mag/split/time/paper/train.csv.gz',
             val_file = r'/media/aslan/50E4BE16E4BDFDF2/DATA/CODE/new_project/datasets/ogbn-mag/ogbn_mag/split/time/paper/valid.csv.gz',
             test_file = r'/media/aslan/50E4BE16E4BDFDF2/DATA/CODE/new_project/datasets/ogbn-mag/ogbn_mag/split/time/paper/test.csv.gz'):
    data = torch.load(path)

    edge_data = data[0]['edge_index_dict']

    num_nodes_dict = data[0]['num_nodes_dict']
    num_paper = num_nodes_dict['paper']
    num_author = num_nodes_dict['author']
    num_subject = num_nodes_dict['field_of_study']
    num_institution = num_nodes_dict['institution']

    paper_data = data[0]['x_dict']['paper'].numpy()
    y_data = data[0]['y_dict']['paper'].numpy()

    idx0, idx1 = np.where(y_data < 4)

    # smple
    samples_paper = idx0

    author_paper = edge_data[('author', 'writes', 'paper')].numpy()
    row = author_paper[0, :]
    col = author_paper[1, :]
    data = np.ones_like(row)
    author_paper_csr_matrix = coo_matrix((data, (row, col)), shape=(num_author, num_paper)).tocsr()

    samples_author = author_paper_csr_matrix[:, samples_paper]
    samples_author = samples_author.sum(axis=-1)
    idx0, idx1 = np.where(samples_author > 0)
    samples_author = idx0

    samples_institution = np.random.choice(np.arange(0, num_institution),
                                           int(num_institution * 0.1))
    #
    # samples_paper = np.random.choice(np.arange(0,len(num_paper)),
    #                                  int(len(num_paper)*0.1))

    # author_institution_author = edge_data[('author', 'affiliated_with', 'institution')].numpy()
    # row = author_institution_author[0, :]
    # col = author_institution_author[1, :]
    # ones = np.ones_like(row)
    # author_institution_csr_matrix = coo_matrix((ones, (row, col)),
    #                                 shape=(num_author, num_institution)).tocsr()[samples_paper,:]
    # author_institution_author = author_institution_csr_matrix.dot(author_institution_csr_matrix.transpose())

    paper_data = paper_data[samples_paper]
    y_data = y_data[samples_paper]

    # node_year = data[0]['node_year']
    # edge_reltype = data[0]['edge_reltype']

    paper_subject = edge_data[('paper', 'has_topic', 'field_of_study')].numpy()
    row = paper_subject[0, :]
    col = paper_subject[1, :]
    data = np.ones_like(row)
    paper_subject_csr_matrix = coo_matrix((data, (row, col)), shape=(num_paper, num_subject)).tocsr()
    # paper_subject_paper = paper_subject_csr_matrix.dot(paper_subject_csr_matrix.transpose())
    paper_subject_csr_matrix = paper_subject_csr_matrix[samples_paper]

    paper_cite_paper = edge_data[('paper', 'cites', 'paper')].numpy()
    row = paper_cite_paper[0, :]
    col = paper_cite_paper[1, :]
    data = np.ones_like(row)
    paper_cite_paper_csr_matrix = coo_matrix((data, (row, col)), shape=(num_paper, num_paper)).tocsr()
    paper_cite_paper_csr_matrix = paper_cite_paper_csr_matrix[samples_paper, :]
    paper_cite_paper_csr_matrix = paper_cite_paper_csr_matrix[:, samples_paper]

    author_paper = edge_data[('author', 'writes', 'paper')].numpy()
    row = author_paper[0, :]
    col = author_paper[1, :]
    data = np.ones_like(row)
    author_paper_csr_matrix = coo_matrix((data, (row, col)), shape=(num_author, num_paper)).tocsr()
    author_paper_csr_matrix = author_paper_csr_matrix[samples_author, :]
    author_paper_csr_matrix = author_paper_csr_matrix[:, samples_paper]

    author_institution_author = edge_data[('author', 'affiliated_with', 'institution')].numpy()
    row = author_institution_author[0, :]
    col = author_institution_author[1, :]
    data = np.ones_like(row)
    author_institution_csr_matrix = coo_matrix((data, (row, col)), shape=(num_author, num_institution)).tocsr()
    author_institution_csr_matrix = author_institution_csr_matrix[samples_author]
    author_institution_csr_matrix = author_institution_csr_matrix[:, samples_institution]
    author_institution_author = author_institution_csr_matrix.dot(author_institution_csr_matrix.transpose())

    # feature_authors = paper_data.dot(author_paper_csr_matrix.transpose())
    feature_authors = np.zeros([samples_author.shape[0], paper_data.shape[1]])
    # nums = paper_data.shape[0]//100
    for i in range(samples_author.shape[0]):
        indexs = author_paper_csr_matrix[i].nonzero()
        for j in range(len(indexs[0])):
            row = indexs[0][j]
            col = indexs[1][j]
            feature_authors[i] += paper_data[col]
        # temp_feature = author_paper_csr_matrix.dot(paper_data[i].reshape(1,-1))
        # feature_authors = feature_authors+ temp_feature.todense()
        feature_authors[i] /= len(indexs)

    train_index = read_gz(tain_file)
    train_index = np.array(train_index)
    train_index = np.intersect1d(train_index, samples_paper)
    # indexes = samples_paper.argwhere(train_index)
    indexes = []
    for i in range(train_index.shape[0]):
        indexes.append(np.where(samples_paper == train_index[i])[0])
    train_index = np.concatenate(indexes)

    val_index = read_gz(val_file)
    val_index = np.array(val_index)
    val_index = np.intersect1d(val_index, samples_paper)
    indexes = []
    for i in range(val_index.shape[0]):
        indexes.append(np.where(samples_paper == val_index[i])[0])
    val_index = np.concatenate(indexes)

    test_index = read_gz(test_file)
    test_index = np.array(test_index)
    test_index = np.intersect1d(test_index, samples_paper)
    indexes = []
    for i in range(test_index.shape[0]):
        indexes.append(np.where(samples_paper == test_index[i])[0])
    test_index = np.concatenate(indexes)


    savemat(r'../exp/data/ogbn_mag_graph.mat', {
                                                'AFA':author_institution_author,
                                                'PS':paper_subject_csr_matrix,
                                                'PCP':paper_cite_paper_csr_matrix,
                                                'PA':author_paper_csr_matrix,
                                                'PT':paper_data,
                                                'AT':feature_authors,

                                                'train_idx':train_index,
                                                'val_idx':val_index,
                                                'test_idx':test_index,

                                                'PL':y_data})


if __name__=="__main__":
    path = r'/media/aslan/50E4BE16E4BDFDF2/DATA/CODE/new_project/datasets/ogbn-mag/ogbn_mag/processed/geometric_data_processed.pt'
    propcess(path)

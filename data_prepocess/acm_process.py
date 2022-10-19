from scipy.io import loadmat,savemat
import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
import random


def load_acm_redict(dir_path):
    SEED = 123
    np.random.seed(SEED)
    random.seed(SEED)

    mat_file = loadmat(dir_path)

    paper_author = mat_file['PvsA'].todense()
    author_sum = paper_author.sum(axis=1)
    non_zeros_index = np.argwhere(author_sum > 0)[:, 0].tolist()
    # get target paper idx
    paper_conf = mat_file['PvsC'].nonzero()[1]  # [non_zeros_index[:,0]].nonzero()[1]  # paper to conference
    # DataBase
    paper_db = np.isin(paper_conf, [1, 13])  # 1 or 13
    paper_db_idx = np.where(paper_db == True)[0]  # total 1994 papers
    paper_db_idx = list(set(paper_db_idx.tolist()).intersection(set(non_zeros_index)))
    paper_db_idx = np.sort(np.random.choice(paper_db_idx, 994, replace=False))  # random choice 994
    # Data Mining
    paper_dm = np.isin(paper_conf, [0])  # 0
    paper_dm_idx = np.where(paper_dm == True)[0]
    paper_dm_idx = list(set(paper_dm_idx.tolist()).intersection(set(non_zeros_index)))
    # Wireless Communication
    paper_wc = np.isin(paper_conf, [9, 10])
    paper_wc_idx = np.where(paper_wc == True)[0]
    paper_wc_idx = list(set(paper_wc_idx.tolist()).intersection(set(non_zeros_index)))
    # paper_id
    paper_idx = np.sort(paper_db_idx.tolist() + paper_dm_idx + paper_wc_idx)
    paper_redict = {}
    for i in range(len(paper_idx)):
        paper_redict[ paper_idx[i] ]=i
    # label
    paper_target = []
    target_for_save = []
    for idx in paper_idx:
        if idx in paper_db_idx:
            paper_target.append(0)
            target_for_save.append([1.,0.,0.])
        elif idx in paper_wc_idx:
            paper_target.append(1)
            target_for_save.append([0.,1.,0.])
        else:
            paper_target.append(2)
            target_for_save.append([0.,0.,1.])
    # paper_id -> target
    paper_target = np.array(paper_target)
    target_for_save = np.array(target_for_save)



    tmp = mat_file['PvsL'][paper_idx].todense()
    matrix = tmp.A
    sum_dim = np.sum(matrix, axis=0)
    sum_dim_indx = np.argsort(sum_dim)

    #tmp= tmp[:,sum_dim_indx[:-5]]
    #tmp= tmp[:,sum_dim_indx[::2]]
    A_plp = tmp.dot(tmp.T)
    A_plp = np.where(A_plp>0.5,np.ones_like(A_plp),np.zeros_like(A_plp))
    A_plp = csr_matrix(A_plp)
    #print(A_plp_.sum())

    # graph 0    paper -> subject -> paper
    """
    tmp = mat_file['PvsL'][paper_idx].nonzero()
    papers = tmp[0]
    matrix = mat_file['PvsL'][paper_idx].toarray().T
    A_plp = np.zeros([len(paper_idx), len(paper_idx)])
    for i in range(len(papers)):
        subject_id = tmp[1][i]            # subjec id
        idx =  tmp[0][i]
        for j in range(len(matrix[subject_id, :])):     # subject->paper
            if matrix[subject_id, j]>0:
                idy = j
                A_plp[idx, idy] += matrix[subject_id, j]
    # check A_ptp is right
    A_plp_t = A_plp.T
    error = np.abs(A_plp - A_plp_t)
    assert np.sum(error) == 0
    A_plp = np.where(A_plp > 0, np.ones_like(A_plp), np.zeros_like(A_plp))
    """


    # graph 0    author -> stitute -> author


    authors_idx = mat_file['PvsA'][paper_idx].nonzero()
    authors_idx = np.unique(authors_idx[1])
    authors_idx = np.sort(authors_idx)
    authors_redict = {}
    for i in range(len(authors_idx)):
        authors_redict[authors_idx[i]] = i

    tmp = mat_file['AvsF'][authors_idx].todense()
    matrix = tmp.A
    sum_dim = np.sum(matrix, axis=0)
    sum_dim_indx = np.argsort(sum_dim)

    #tmp= tmp[:,sum_dim_indx[:-100]]
    #tmp= tmp[:,sum_dim_indx[::2]]
    A_afa = tmp.dot(tmp.T)
    A_afa = np.where(A_afa>0.5,np.ones_like(A_afa),np.zeros_like(A_afa))
    A_afa = csr_matrix(A_afa)
    """
    tmp = mat_file['AvsF'][authors_idx].nonzero()
    institue_author = mat_file['AvsF'][authors_idx].toarray().T
    A_afa = np.zeros([len(authors_idx), len(authors_idx)])
    for i in range(len(tmp[0])):
        institue_id = tmp[1][i]
        idx = tmp[0][i]
        for j in range(len(institue_author[institue_id, :])):     # subject->paper
            if institue_author[institue_id, j]>0:
                idy = j
                A_afa[idx, idy] += institue_author[institue_id, j]

    A_afa_t = A_afa.T
    error = np.abs(A_afa - A_afa_t)
    assert np.sum(error) == 0
    A_afa = np.where(A_afa > 0, np.ones_like(A_afa), np.zeros_like(A_afa))
    """

    # paper -> author
    # A_pa = np.zeros([len(paper_idx), len(authors_idx)])
    # paper_author = mat_file['PvsA'][paper_idx].nonzero()
    # for i in range(len(paper_author[0])):
    #     paper_redict_id =paper_author[0][i]
    #     author_id =paper_author[1][i]
    #     author_redict_id = authors_redict[author_id] #np.where(authors_idx==author_id)[0]
    #     A_pa[paper_redict_id,author_redict_id] = 1.0

    # paper_author = mat_file['PvsA'][paper_idx[2883],:].nonzero()
    paper_author = mat_file['PvsA'][paper_idx,:][:,authors_idx]
    A_pa = paper_author #csr_matrix(A_pa)

    # paper feature
    terms_idx = mat_file['TvsP'].transpose()[paper_idx].nonzero()[1]
    terms_idx = np.unique(np.array(terms_idx))
    terms_idx = np.sort(terms_idx)
    terms_redict = {}
    for i in range(len(terms_idx)):
        terms_redict[terms_idx[i]] = i
    tmp = mat_file['TvsP'].transpose()[paper_idx].nonzero()
    paper_features = np.zeros([len(paper_idx),len(terms_idx)])
    for i in range(len(tmp[0])):
        term_id = tmp[1][i]
        #paper_id = tmp[0][i]
        idx = tmp[0][i]
        idy = terms_redict[term_id] #np.where(terms_idx ==term_id)[0]  # paper id ->adj id
        paper_features[idx, idy] += 1.0

    # author feature
    feature_paper = mat_file['TvsP']
    papers_author = mat_file['PvsA']
    feature_authors = feature_paper.dot(papers_author)
    author_features = feature_authors.transpose()[authors_idx].toarray()
    author_featuresx = author_features[:,terms_idx]

    feature_paper = feature_paper.toarray()
    feature_authors = np.zeros([papers_author.shape[1],feature_paper.shape[0]])
    #nums = paper_data.shape[0]//100
    for i in range(papers_author.shape[1]):
        indexs = papers_author[:,i].nonzero()
        for j in range(len(indexs[0])):
            row = indexs[0][j]
            col = indexs[1][j]
            feature_authors[i] += feature_paper[:,row]
        #temp_feature = author_paper_csr_matrix.dot(paper_data[i].reshape(1,-1))
        #feature_authors = feature_authors+ temp_feature.todense()
        feature_authors[i] /= len(indexs)
    author_features = feature_authors[authors_idx,:]
    author_features = author_features[:,terms_idx]


    #split train val test
    # Train, Valid
    train_valid_DB = list(np.random.choice(np.where(paper_target == 0)[0], 300, replace=False))
    train_valid_WC = list(np.random.choice(np.where(paper_target == 1)[0], 300, replace=False))
    train_valid_DM = list(np.random.choice(np.where(paper_target == 2)[0], 300, replace=False))

    train_idx = np.array(train_valid_DB[:200] + train_valid_WC[:200] + train_valid_DM[:200])
    train_target = paper_target[train_idx]
    # train_label = np.vstack((train_idx, train_target)).transpose()
    valid_idx = np.array(train_valid_DB[200:] + train_valid_WC[200:] + train_valid_DM[200:])
    valid_target = paper_target[valid_idx]
    #valid_label = np.vstack((valid_idx, valid_target)).transpose()
    test_idx = np.array(list((set(np.arange(paper_target.shape[0])) - set(train_idx)) - set(valid_idx)))
    test_target = paper_target[test_idx]
    # test_label = np.vstack((test_idx, test_target)).transpose()

    savemat(r'../exp/acm/data/ACM_multi_graph.mat', {'PSP':A_plp,
                                                'AFA':A_afa,
                                                'PA':A_pa,
                                                'PT':paper_features,
                                                'AT':author_features,
                                                'train_idx':train_idx,
                                                'train_taget':train_target,
                                                'val_idx':valid_target,
                                                'val_taget':valid_idx,
                                                'test_idx':test_idx,
                                                'test_taget':test_target,
                                                     'PL':target_for_save})

    print("finish")

if __name__=="__main__":
    acm3025_path = r'/media/aslan/50E4BE16E4BDFDF2/DATA/CODE/new_project/datasets/acm/ACM3025.mat'
    acm_path = r'/media/aslan/50E4BE16E4BDFDF2/DATA/CODE/new_project/datasets/acm/ACM.mat'
    #data3025 = loadmat(acm3025_path)
    #data = loadmat(acm_path)
    load_acm_redict(acm_path)

    # paper -> subject -> paper
    # author -> F  -> author
    #

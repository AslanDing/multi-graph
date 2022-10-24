import os
import numpy as np
import scipy.sparse
from torch_geometric.data import HeteroData,DataLoader,Dataset
from scipy.io import loadmat,savemat
from  scipy.sparse import coo_matrix
import torch
import torch_geometric.transforms as T

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

data_dir = r'../exp/pubmed/data'
data_path = data_dir + r'/PubMed_process.mat'

def label_to_vector(index,len=8):
    vec = np.zeros([1,len])
    vec[0,index]=1.0
    return vec

class PubMedDataset(Dataset):
    def __init__(self,mode='train',k = [16,8],k1 = [16,8],
                 root=data_dir,
                 data_p = data_path,transform=None,
                 pre_transform=None, pre_filter=None,
                 generate = False):
        self.generate = generate
        self.root = root

        self.y_dim = 8  # self.Paper_Label.shape[1]

        self.GENE_dim = 200 #self.feature0.shape[1]
        self.DISEASE_dim = 200 #self.feature1.shape[1]
        self.CHEMICAL_dim = 200 #self.feature2.shape[1]
        self.SPECIES_dim = 200 #self.feature3.shape[1]

        self.k = len(k)
        self.k_list = k
        self.k1_list = k1
        if mode in ['train', 'test', 'val']:
            self.mode = mode
        else:
            raise ValueError("mode only support  train test val")


        if not generate:
            lists = torch.load(os.path.join(root, "list_for_item.pt"))
            self.train_list = lists[0]
            self.test_list = lists[1]

        else:

            datas = loadmat(data_p)

            self.feature0 = datas['NF0']
            self.feature1 = datas['NF1']
            self.feature2 = datas['NF2']
            self.feature3 = datas['NF3']

            self.A00 = datas['A00']
            self.feature0_degrees = self.A00.sum(axis=1)
            self.A11 = datas['A11']
            self.feature1_degrees = self.A11.sum(axis=1)
            self.A22 = datas['A22']
            self.feature2_degrees = self.A22.sum(axis=1)
            self.A33 = datas['A33']
            self.feature3_degrees = self.A33.sum(axis=1)

            self.A01 = datas['A01']
            self.A01_feature0_degrees = self.A01.sum(axis=1)
            self.A01_feature1_degrees = self.A01.sum(axis=0)
            self.A20 = datas['A20']
            self.A20_feature2_degrees = self.A20.sum(axis=1)
            self.A20_feature0_degrees = self.A20.sum(axis=0)
            self.A30 = datas['A30']
            self.A30_feature3_degrees = self.A30.sum(axis=1)
            self.A30_feature0_degrees = self.A30.sum(axis=0)
            self.A21 = datas['A21']
            self.A21_feature2_degrees = self.A21.sum(axis=1)
            self.A21_feature1_degrees = self.A21.sum(axis=0)
            self.A31 = datas['A31']
            self.A31_feature3_degrees = self.A31.sum(axis=1)
            self.A31_feature1_degrees = self.A31.sum(axis=0)
            self.A23 = datas['A23']
            self.A23_feature2_degrees = self.A23.sum(axis=1)
            self.A23_feature3_degrees = self.A23.sum(axis=0)

            self.train_list = datas['train_list']
            self.train_label = datas['train_label']

            self.test_list = datas['test_list']
            self.test_label = datas['test_label']

            del datas

            data = HeteroData()
            data['domain0'].x = torch.from_numpy(self.feature0).float()
            data['domain1'].x = torch.from_numpy(self.feature1).float()
            data['domain2'].x = torch.from_numpy(self.feature2).float()
            data['domain3'].x = torch.from_numpy(self.feature3).float()

            self.data = data

            # self.data = T.NormalizeFeatures()(self.data)
            # self.transform_0 = T.ToUndirected()
            # self.transform_1 = T.AddSelfLoops()



            self.domain_cross_sample = 16

        super(PubMedDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt']

    def download(self):
        pass

    def process(self):
        if not self.generate:
            return
        lists = [self.train_list,self.test_list]
        label_lists = [self.train_label,self.test_label]
        save_names = ["train","test"]

        torch.save(lists,os.path.join(self.root, "list_for_item.pt"))

        for ilist in range(len(lists)):
            list_for_iter = lists[ilist]
            label_for_iter = label_lists[ilist]
            name_for_save = save_names[ilist]
            for item in range(list_for_iter.shape[1]):
                id = item % list_for_iter.shape[1]
                idx = list_for_iter[0, id]
                label = label_for_iter[0, id]

                label = label_to_vector(label)
                list_node_index = [idx]
                select = [idx]
                # edge_indx = []
                for i in range(self.k):
                    select_list = []
                    for ii in select:
                        indexs = self.A11[ii].nonzero()
                        select_tmp = []
                        for j in range(len(indexs[0])):
                            row = indexs[0][j]
                            col = indexs[1][j]
                            select_tmp.append(col)

                        if len(select_tmp) <= 0:
                            continue
                        elif len(select_tmp) < self.k_list[i]:
                            select_list.extend(select_tmp)
                            continue

                        degrees = self.feature1_degrees[select_tmp]
                        degreelist = np.argsort(degrees[:, 0], axis=0)
                        # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                        args = degreelist[-self.k_list[i]:, 0].A[:, 0].tolist()
                        select_tmp = [select_tmp[i] for i in args]
                        # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                        # select_tmp = select_tmp[:self.k_list[i]]
                        select_list.extend(select_tmp)
                        # for ix in select_tmp:
                        #     edge_indx.append([ix, ii])

                    select_list = list(set(select_list) - set(list_node_index))
                    select = select_list
                    list_node_index.extend(select_list)

                # if len(edge_indx) > 1:
                #     edge_indx = np.array(edge_indx)
                #     edge_matrix = scipy.sparse.coo_matrix((np.ones(edge_indx.shape[0]), (edge_indx[:, 0], edge_indx[:, 1])),
                #                                           shape=(self.A11.shape[0], self.A11.shape[0]))
                # else:
                #     edge_matrix = 1

                domain1_idx = list_node_index
                index = np.argwhere(domain1_idx == idx)
                # print(len(domain1_idx),end=" ")
                # A11sub = self.A11*edge_matrix
                A11sub = self.A11[ domain1_idx,:]
                A11sub = A11sub[:,domain1_idx].nonzero()

                # domain 0
                idx_domain0 = self.A01[:, idx].nonzero()
                idx_domain0 = idx_domain0[0].tolist()
                if len(idx_domain0) < self.domain_cross_sample:
                    select_idx_tmp = idx_domain0
                else:
                    # select_idx_tmp = np.random.choice(idx_domain0, self.domain_cross_sample, replace=False).tolist()
                    degrees = self.A01_feature0_degrees[idx_domain0]
                    degreelist = np.argsort(degrees[:, 0], axis=0)
                    # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                    args = degreelist[-self.domain_cross_sample:, 0].A[:, 0].tolist()
                    select_idx_tmp = [idx_domain0[i] for i in args]

                    # select_idx_tmp = idx_domain0[:self.domain_cross_sample]

                list_node_index = select_idx_tmp
                select = select_idx_tmp
                # edge_indx = []
                for i in range(self.k):
                    select_list = []
                    for ii in select:
                        indexs = self.A00[ii].nonzero()
                        select_tmp = []
                        for j in range(len(indexs[0])):
                            row = indexs[0][j]
                            col = indexs[1][j]
                            select_tmp.append(col)

                        if len(select_tmp) <= 0:
                            continue
                        elif len(select_tmp) < self.k_list[i]:
                            select_list.extend(select_tmp)
                            continue

                        degrees = self.feature0_degrees[select_tmp]
                        degreelist = np.argsort(degrees[:, 0], axis=0)
                        # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                        args = degreelist[-self.k_list[i]:, 0].A[:, 0].tolist()
                        select_tmp = [select_tmp[i] for i in args]
                        select_list.extend(select_tmp)
                        # for ix in select_tmp:
                        #     edge_indx.append([ix, ii])

                    select_list = list(set(select_list) - set(list_node_index))
                    select = select_list
                    list_node_index.extend(select_list)

                # if len(edge_indx) > 1:
                #     edge_indx = np.array(edge_indx)
                #     edge_matrix = scipy.sparse.coo_matrix((np.ones(edge_indx.shape[0]), (edge_indx[:, 0], edge_indx[:, 1])),
                #                                           shape=(self.A00.shape[0], self.A00.shape[0]))
                # else:
                #     edge_matrix = 1

                domain0_idx = list_node_index
                # print(len(domain0_idx),end=" ")
                A00sub = self.A00[ domain0_idx,:]
                A00sub = A00sub[:,domain0_idx].nonzero()

                # domain 2
                idx_domain2 = self.A21[:, idx].nonzero()
                idx_domain2 = idx_domain2[0].tolist()
                if len(idx_domain2) < self.domain_cross_sample:
                    select_idx_tmp = idx_domain2
                else:
                    # select_idx_tmp = np.random.choice(idx_domain2, self.domain_cross_sample, replace=False).tolist()
                    degrees = self.A21_feature2_degrees[idx_domain2]
                    degreelist = np.argsort(degrees[:, 0], axis=0)
                    # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                    args = degreelist[-self.domain_cross_sample:, 0].A[:, 0].tolist()
                    select_idx_tmp = [idx_domain2[i] for i in args]

                list_node_index = select_idx_tmp
                select = select_idx_tmp
                # edge_indx = []
                for i in range(self.k):
                    select_list = []
                    for ii in select:
                        indexs = self.A22[ii].nonzero()
                        select_tmp = []
                        for j in range(len(indexs[0])):
                            row = indexs[0][j]
                            col = indexs[1][j]
                            select_tmp.append(col)
                        if len(select_tmp) <= 0:
                            continue
                        elif len(select_tmp) < self.k_list[i]:
                            select_list.extend(select_tmp)
                            continue
                        # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                        degrees = self.feature2_degrees[select_tmp]
                        degreelist = np.argsort(degrees[:, 0], axis=0)
                        # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                        args = degreelist[-self.k_list[i]:, 0].A[:, 0].tolist()
                        select_tmp = [select_tmp[i] for i in args]
                        # select_tmp = select_tmp[:self.k_list[i]]
                        select_list.extend(select_tmp)
                        # for ix in select_tmp:
                        #     edge_indx.append([ix, ii])
                    select_list = list(set(select_list) - set(list_node_index))
                    select = select_list
                    list_node_index.extend(select_list)

                # if len(edge_indx) > 1:
                #     edge_indx = np.array(edge_indx)
                #     edge_matrix = scipy.sparse.coo_matrix((np.ones(edge_indx.shape[0]), (edge_indx[:, 0], edge_indx[:, 1])),
                #                                           shape=(self.A22.shape[0], self.A22.shape[0]))
                # else:
                #     edge_matrix = 1
                domain2_idx = list_node_index
                # print(len(domain2_idx),end=" ")
                A22sub = self.A22[domain2_idx,:]
                A22sub = A22sub[:,domain2_idx].nonzero()

                # domain 3
                idx_domain3 = self.A31[:, idx].nonzero()
                idx_domain3 = idx_domain3[0].tolist()
                if len(idx_domain3) < self.domain_cross_sample:
                    select_idx_tmp = idx_domain3
                else:
                    # select_idx_tmp = np.random.choice(idx_domain3, self.domain_cross_sample, replace=False).tolist()
                    degrees = self.A31_feature3_degrees[idx_domain3]
                    degreelist = np.argsort(degrees[:, 0], axis=0)
                    # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                    args = degreelist[-self.domain_cross_sample:, 0].A[:, 0].tolist()
                    select_idx_tmp = [idx_domain3[i] for i in args]
                    # select_idx_tmp = idx_domain3[:self.domain_cross_sample]
                list_node_index = select_idx_tmp
                select = select_idx_tmp
                # edge_indx = []
                for i in range(self.k):
                    select_list = []
                    for ii in select:
                        indexs = self.A33[ii].nonzero()
                        select_tmp = []
                        for j in range(len(indexs[0])):
                            row = indexs[0][j]
                            col = indexs[1][j]
                            select_tmp.append(col)
                        if len(select_tmp) <= 0:
                            continue
                        elif len(select_tmp) < self.k_list[i]:
                            select_list.extend(select_tmp)
                            continue
                        # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                        degrees = self.feature3_degrees[select_tmp]
                        degreelist = np.argsort(degrees[:, 0], axis=0)
                        # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                        args = degreelist[-self.k_list[i]:, 0].A[:, 0].tolist()
                        select_tmp = [select_tmp[i] for i in args]
                        # select_tmp = select_tmp[:self.k_list[i]]
                        select_list.extend(select_tmp)
                        # for ix in select_tmp:
                        #     edge_indx.append([ix, ii])
                    select_list = list(set(select_list) - set(list_node_index))
                    select = select_list
                    list_node_index.extend(select_list)

                # if len(edge_indx) > 1:
                #     edge_indx = np.array(edge_indx)
                #     edge_matrix = scipy.sparse.coo_matrix((np.ones(edge_indx.shape[0]), (edge_indx[:, 0], edge_indx[:, 1])),
                #                                           shape=(self.A33.shape[0], self.A33.shape[0]))
                # else:
                #     edge_matrix = 1
                domain3_idx = list_node_index
                # print(len(domain3_idx),end=" ")
                A33sub = self.A33[:, domain3_idx]
                A33sub = A33sub[domain3_idx, :].nonzero()

                # layers
                A01sub = self.A01[domain0_idx, :]
                A01sub = A01sub[:, domain1_idx].nonzero()
                A12sub = self.A21.transpose()[:, domain2_idx]
                A12sub = A12sub[domain1_idx, :].nonzero()
                A13sub = self.A31.transpose()[:, domain3_idx]
                A13sub = A13sub[domain1_idx, :].nonzero()

                data = self.data.clone()

                data['domain0'].x = data['domain0'].x[domain0_idx, :]
                data['domain0']['mask'] = torch.tensor(len(domain0_idx)).to(torch.long)
                data['domain1'].x = data['domain1'].x[domain1_idx, :]
                data['domain1']['mask'] = torch.tensor(index[0, 0]).to(torch.long)
                data['domain2'].x = data['domain2'].x[domain2_idx, :]
                data['domain2']['mask'] = torch.tensor(len(domain2_idx)).to(torch.long)
                data['domain3'].x = data['domain3'].x[domain3_idx, :]
                data['domain3']['mask'] = torch.tensor(len(domain3_idx)).to(torch.long)

                data['domain0', 'to', 'domain0'].edge_index = torch.stack([torch.from_numpy(A00sub[0]),
                                                                           torch.from_numpy(A00sub[1])], dim=0)
                data['domain0', 'to', 'domain0']['mask'] = torch.tensor(A00sub[0].shape[0]).to(torch.long)

                data['domain1', 'to', 'domain1'].edge_index = torch.stack([torch.from_numpy(A11sub[0]),
                                                                           torch.from_numpy(A11sub[1])], dim=0)
                data['domain1', 'to', 'domain1']['mask'] = torch.tensor(A11sub[0].shape[0]).to(torch.long)

                data['domain2', 'to', 'domain2'].edge_index = torch.stack([torch.from_numpy(A22sub[0]),
                                                                           torch.from_numpy(A22sub[1])], dim=0)
                data['domain2', 'to', 'domain2']['mask'] = torch.tensor(A22sub[0].shape[0]).to(torch.long)

                data['domain3', 'to', 'domain3'].edge_index = torch.stack([torch.from_numpy(A33sub[0]),
                                                                           torch.from_numpy(A33sub[1])], dim=0)
                data['domain3', 'to', 'domain3']['mask'] = torch.tensor(A33sub[0].shape[0]).to(torch.long)

                data['domain1', 'to', 'domain0'].edge_index = torch.stack([torch.from_numpy(A01sub[1]),
                                                                           torch.from_numpy(A01sub[0])], dim=0)
                data['domain1', 'to', 'domain0']['mask'] = torch.tensor(A01sub[0].shape[0]).to(torch.long)

                data['domain0', 'to', 'domain1'].edge_index = torch.stack([torch.from_numpy(A01sub[0]),
                                                                           torch.from_numpy(A01sub[1])], dim=0)
                data['domain0', 'to', 'domain1']['mask'] = torch.tensor(A01sub[0].shape[0]).to(torch.long)

                data['domain1', 'to', 'domain2'].edge_index = torch.stack([torch.from_numpy(A12sub[0]),
                                                                           torch.from_numpy(A12sub[1])], dim=0)
                data['domain1', 'to', 'domain2']['mask'] = torch.tensor(A12sub[0].shape[0]).to(torch.long)
                data['domain2', 'to', 'domain1'].edge_index = torch.stack([torch.from_numpy(A12sub[1]),
                                                                           torch.from_numpy(A12sub[0])], dim=0)
                data['domain2', 'to', 'domain1']['mask'] = torch.tensor(A12sub[0].shape[0]).to(torch.long)

                data['domain1', 'to', 'domain3'].edge_index = torch.stack([torch.from_numpy(A13sub[0]),
                                                                           torch.from_numpy(A13sub[1])], dim=0)
                data['domain1', 'to', 'domain3']['mask'] = torch.tensor(A13sub[0].shape[0]).to(torch.long)
                data['domain3', 'to', 'domain1'].edge_index = torch.stack([torch.from_numpy(A13sub[1]),
                                                                           torch.from_numpy(A13sub[0])], dim=0)
                data['domain3', 'to', 'domain1']['mask'] = torch.tensor(A13sub[0].shape[0]).to(torch.long)

                data['domain1'].y = torch.from_numpy(label.reshape([1, -1])).float()

                torch.save(data,os.path.join(self.processed_dir,name_for_save+f"_data_{item}.pt"))

    def len(self):
        if self.mode == 'train':
            return self.train_list.shape[1]
        elif self.mode == 'test':
            return self.test_list.shape[1]
        elif self.mode == 'val':
            return self.train_list.shape[1]

    def get(self, item):
        #print("item",item)  dim 1
        if self.mode == 'train':
            id = item % self.train_list.shape[1]
            data = torch.load(os.path.join(self.processed_dir,"train"+f"_data_{id}.pt"))

        elif self.mode == 'test':
            id = item % self.test_list.shape[1]
            data = torch.load(os.path.join(self.processed_dir,"test"+f"_data_{id}.pt"))
        elif self.mode == 'val':
            id = item % self.test_list.shape[1]
            data = torch.load(os.path.join(self.processed_dir,"test"+f"_data_{id}.pt"))

        return data

if __name__=="__main__":
    import random

    seed = 10
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    train_dataset =  PubMedDataset('train',generate=True)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # paper_dim = train_dataset.paper_dim
    # author_dim = train_dataset.author_dim
    # label_dim = train_dataset.y_dim

    for index, data in enumerate(train_dataloader):
        print("index ",index)  # ,end="  "


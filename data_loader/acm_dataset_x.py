import numpy as np
from torch_geometric.data import HeteroData,DataLoader,Dataset
from scipy.io import loadmat,savemat
from  scipy.sparse import coo_matrix
import torch
import torch_geometric.transforms as T

def label_to_vector(index):
    vec = np.zeros([1,3])
    vec[0,index]=1.0
    return vec

data_dir = r'../exp/acm'
data_path = data_dir + r'/ACM_multi_graph.mat'
class ACMDataset(Dataset):
    def __init__(self, mode='train',k = [16,8],k1 = [16,8],
                 root=data_dir,data_p = data_path, transform=None,
                 pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.k = len(k)
        self.k_list = k
        self.k1_list = k1
        if mode in ['train','test','val']:
            self.mode = mode
        else:
            raise ValueError("mode only support  train test val")

        datas = loadmat(data_p)
        # count = len(datas['PSP'].nonzero()[0])+len(datas['AFA'].nonzero()[0])+len(datas['PA'].nonzero()[0])
        self.Paper_Subject_Paper = datas['PSP']
        self.paper_degrees = self.Paper_Subject_Paper.sum(axis=1)
        self.Author_Institute_Author = datas['AFA']
        self.author_degrees = self.Author_Institute_Author.sum(axis=1)

        self.Paper_Author = datas['PA']
        self.PA_paper_degrees = self.Paper_Author.sum(axis=1)
        self.PA_author_degrees = self.Paper_Author.sum(axis=0)

        self.Paper_Features = datas['PT']
        self.Paper_Label = datas['PL']
        self.Author_Features = datas['AT']

        self.train_idx = datas['train_idx']
        self.train_label = datas['train_taget']

        self.val_idx = datas['val_idx']
        self.val_label = datas['val_taget']

        self.test_idx = datas['test_idx']
        self.test_label = datas['test_taget']

        del datas

        data = HeteroData()
        data['paper'].x = torch.from_numpy(self.Paper_Features).float()
        # data['paper'].y = self.Paper_Label
        data['author'].x = torch.from_numpy(self.Author_Features).float()

        # data['paper', 'subject', 'paper'].edge_index = coo_matrix(Paper_Subject_Paper)
        # data['author', 'Institute', 'author'].edge_index = coo_matrix(Author_Institute_Author)
        # data['paper', 'from', 'author'].edge_index = coo_matrix(Paper_Author)

        self.data = data

        self.data = T.NormalizeFeatures()(self.data)
        # self.transform_0 = T.ToUndirected()
        # self.transform_1 = T.AddSelfLoops()

        self.paper_dim = self.Paper_Features.shape[1]
        self.author_dim = self.Author_Features.shape[1]
        self.y_dim = self.Paper_Label.shape[1]

        self.paper_len = self.Paper_Subject_Paper.shape[0]

        self.domain_cross_sample = 16

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt']

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        if self.mode == 'train':
            return self.train_idx.shape[1]
        elif self.mode == 'test':
            return self.test_idx.shape[1]
        elif self.mode == 'val':
            return self.val_idx.shape[1]

    def get(self, item):

        if self.mode == 'train':
            id = item % self.train_idx.shape[1]
            idx = self.train_idx[0,id]
            label = self.train_label[0,id]
        elif self.mode == 'test':
            id = item % self.test_idx.shape[1]
            idx = self.test_idx[0,id]
            label = self.test_label[0,id]
        elif self.mode == 'val':
            id = item % self.val_idx.shape[1]
            idx = self.val_idx[0,id]
            label = self.val_label[0,id]

        #print("item",idx,end=" ")

        # k-hop paper graph
        # paper_vector = np.zeros([1, self.paper_len])
        label = label_to_vector(label)
        # sample
        list_node_index = [idx]
        select = [idx]
        for i in range(self.k):
            select_list = []
            for ii in select:
                indexs = self.Paper_Subject_Paper[ii].nonzero()
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

                degrees = self.paper_degrees[select_tmp]
                degreelist = np.argsort(degrees[:, 0], axis=0)
                # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                args = degreelist[-self.k_list[i]:, 0].A[:, 0].tolist()
                select_tmp = [select_tmp[i] for i in args]
                # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                # select_tmp = select_tmp[:self.k_list[i]]
                select_list.extend(select_tmp)

            select_list = list(set(select_list)-set(list_node_index))
            select = select_list
            list_node_index.extend(select_list)

        paper_vector = list_node_index
        index = np.argwhere(paper_vector == idx)
        # print(len(domain1_idx),end=" ")
        Paper_Subject_Paper_sub = self.Paper_Subject_Paper[paper_vector, :]
        Paper_Subject_Paper_sub = Paper_Subject_Paper_sub[:, paper_vector].nonzero()


        # domain 0
        idx_domain0 = self.Paper_Author[idx,:].nonzero()
        idx_domain0 = idx_domain0[1].tolist()
        if len(idx_domain0)<self.domain_cross_sample:
            select_idx_tmp = idx_domain0
        else:
            # select_idx_tmp = np.random.choice(idx_domain0, self.domain_cross_sample, replace=False).tolist()
            degrees = self.PA_author_degrees[0,idx_domain0]
            degreelist = np.argsort(degrees[:, 0], axis=0)
            # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
            args = degreelist[-self.domain_cross_sample:, 0].A[:, 0].tolist()
            select_idx_tmp = [idx_domain0[i] for i in args]

            #select_idx_tmp = idx_domain0[:self.domain_cross_sample]

        list_node_index = select_idx_tmp
        select = select_idx_tmp
        for i in range(self.k):
            select_list = []
            for ii in select:
                indexs = self.Author_Institute_Author[ii].nonzero()
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

                degrees = self.author_degrees[select_tmp]
                degreelist = np.argsort(degrees[:, 0], axis=0)
                # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                args = degreelist[-self.k_list[i]:, 0].A[:, 0].tolist()
                select_tmp = [select_tmp[i] for i in args]
                select_list.extend(select_tmp)
            select_list = list(set(select_list) - set(list_node_index))
            select = select_list
            list_node_index.extend(select_list)

        author_idx = list_node_index
        # print(len(domain0_idx),end=" ")
        Author_Institute_Author_sub = self.Author_Institute_Author[author_idx, :]
        Author_Institute_Author_sub = Author_Institute_Author_sub[:, author_idx].nonzero()

        Paper_Author = self.Paper_Author[paper_vector,:]
        Paper_Author = Paper_Author[:,author_idx].nonzero()

        data = self.data.clone()
        data['paper'].x = data['paper'].x[paper_vector,:]
        data['paper']['mask'] = torch.tensor(index[0,0]).to(torch.long)
        data['author'].x = data['author'].x[author_idx,:]
        data['author']['mask'] = torch.tensor(len(author_idx)).to(torch.long)

        data['paper', 'subject', 'paper'].edge_index = torch.stack([torch.from_numpy(Paper_Subject_Paper_sub[0]),
                                                                   torch.from_numpy(Paper_Subject_Paper_sub[1])], dim=0)
        data['paper', 'subject', 'paper']['mask'] = torch.tensor(Paper_Subject_Paper_sub[0].shape[0]).to(torch.long)

        data['author', 'Institute', 'author'].edge_index = torch.stack([torch.from_numpy(Author_Institute_Author_sub[0]),
                                                                   torch.from_numpy(Author_Institute_Author_sub[1])], dim=0)
        data['author', 'Institute', 'author']['mask'] = torch.tensor(Author_Institute_Author_sub[0].shape[0]).to(torch.long)

        data['paper','from','author'].edge_index = torch.stack([torch.from_numpy(Paper_Author[0]),
                                                                   torch.from_numpy(Paper_Author[1])], dim=0)
        data['paper','from','author']['mask'] = torch.tensor(Paper_Author[0].shape[0]).to(torch.long)

        data['author','write','paper'].edge_index = torch.stack([torch.from_numpy(Paper_Author[1]),
                                                                   torch.from_numpy(Paper_Author[0])], dim=0)
        data['author','write','paper']['mask'] = torch.tensor(Paper_Author[0].shape[0]).to(torch.long)

        data['paper'].y = torch.from_numpy(label.reshape([1,3])).float()

        return data


if __name__ == "__main__":
    import random

    seed = 10
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = ACMDataset('test')

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    paper_dim = train_dataset.paper_dim
    author_dim = train_dataset.author_dim
    label_dim = train_dataset.y_dim

    for index, data in enumerate(train_dataloader):
        if data['author'].x.shape[0] == 0:
            print(" ")
        print(index,data['author'].x.shape)  # ,end="  "


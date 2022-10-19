import numpy as np
from torch_geometric.data import HeteroData,DataLoader,Dataset
from scipy.io import loadmat,savemat
from  scipy.sparse import coo_matrix
import torch
import torch_geometric.transforms as T

data_dir = r'../exp/ogbn_mag'
data_path = data_dir + r'/ogbn_mag_graph.mat'

def label_to_vector(index):
    vec = np.zeros([1,349])
    vec[0,index]=1.0
    return vec

class OgbnMagDataset(Dataset):
    def __init__(self, mode='train',
                 k = [16,8],
                 k1 = [16,8],
                 root=data_dir,
                 data_p = data_path,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(OgbnMagDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if mode in ['train','test','val']:
            self.mode = mode
        else:
            raise ValueError("mode only support  train test val")
        self.k = len(k)
        self.k_list = k
        self.k1_list = k1


        datas = loadmat(data_p)

        self.Paper_Label = datas['PL']

        self.paper_feature = datas['PT']
        self.author_feature = datas['AT']

        self.paper_cite_paper = datas['PCP']
        self.paper_degrees = self.paper_cite_paper.sum(axis=1)
        self.paper_author = datas['PA']
        self.PA_paper_degrees = self.paper_author.sum(axis=1)
        self.PA_author_degrees = self.paper_author.sum(axis=0)

        # paper subject
        # self.paper_subject = datas['PS']

        self.author_institution = datas['AFA']
        self.AF_author_degrees = self.author_institution.sum(axis=1)
        self.AF_institution_degrees = self.author_institution.sum(axis=0)

        self.train_idx = datas['train_idx']
        self.val_idx = datas['val_idx']
        self.test_idx = datas['test_idx']

        # count_x = (self.author_institution @ (self.author_institution.sum(axis=0).transpose()-1)).sum()/2
        # count = len(self.paper_author.nonzero()[0]) + len(self.paper_cite_paper.nonzero()[0])

        del datas
        data = HeteroData()
        data['paper'].x = torch.from_numpy(self.paper_feature).float()
        # data['paper'].y = self.Paper_Label
        data['author'].x = torch.from_numpy(self.author_feature).float()

        self.data = data
        #self.data = T.NormalizeFeatures()(self.data)
        self.transform_0 = T.ToUndirected()
        self.transform_1 = T.AddSelfLoops()

        self.paper_dim = self.paper_feature.shape[1]
        self.author_dim = self.author_feature.shape[1]
        self.y_dim = 349

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
        # print("item",item)
        if self.mode == 'train':
            id = item % self.train_idx.shape[1]
            idx = self.train_idx[0, id]
            label = self.Paper_Label[idx]
        elif self.mode == 'test':
            id = item % self.test_idx.shape[1]
            idx = self.test_idx[0, id]
            label = self.Paper_Label[idx]
        elif self.mode == 'val':
            id = item % self.val_idx.shape[1]
            idx = self.val_idx[0, id]
            label = self.Paper_Label[idx]

        label = label_to_vector(label)
        # k-hop paper graph
        # paper_vector = np.zeros([1, self.paper_len])

        list_node_index = [idx]
        select = [idx]
        for i in range(self.k):
            select_list = []
            for ii in select:
                indexs = self.paper_cite_paper[:,ii].nonzero()
                select_tmp = []
                for j in range(len(indexs[0])):
                    row = indexs[0][j]
                    col = indexs[1][j]
                    select_tmp.append(row)
                if len(select_tmp)<=0:
                    continue
                elif len(select_tmp) < self.k_list[i]:
                    select_list.extend(select_tmp)
                    continue

                degrees = self.paper_degrees[select_tmp]
                degreelist = np.argsort(degrees[:, 0], axis=0)
                args = degreelist[-self.k_list[i]:, 0].A[:, 0].tolist()
                select_tmp = [select_tmp[i] for i in args]
                #select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                select_list.extend(select_tmp)
            select_list = list(set(select_list) - set(list_node_index))
            select = select_list
            list_node_index.extend(select_list)

        paper_idx= list_node_index
        index = np.argwhere(paper_idx == idx)
        paper2paper = self.paper_cite_paper[paper_idx,:][:,paper_idx].nonzero()

        author_vector = self.paper_author[idx].nonzero()
        idx_lsit = author_vector[1].tolist()
        if len(idx_lsit)<self.domain_cross_sample:
            select_idx_tmp = idx_lsit
        else:
            degrees = self.PA_author_degrees[0,idx_lsit]
            degreelist = np.argsort(degrees[:, 0], axis=0)
            # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
            args = degreelist[-self.domain_cross_sample:, 0].A[:, 0].tolist()
            select_idx_tmp = [idx_lsit[i] for i in args]
            #select_idx_tmp = np.random.choice(idx_lsit, 5, replace=False).tolist()

        list_node_index = select_idx_tmp
        select = select_idx_tmp
        for i in range(self.k):
            select_list = []
            for ii in select:
                indexs = self.author_institution[ii].nonzero()
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

                degrees = self.AF_author_degrees[select_tmp]
                degreelist = np.argsort(degrees[:, 0], axis=0)
                # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                args = degreelist[-self.k_list[i]:, 0].A[:, 0].tolist()
                select_tmp = [select_tmp[i] for i in args]
                select_list.extend(select_tmp)
            select_list = list(set(select_list) - set(list_node_index))
            select = select_list
            list_node_index.extend(select_list)


        author_idx= list_node_index
        author_institution_sub =self.author_institution[author_idx,:]
        author2author = author_institution_sub.dot(author_institution_sub.transpose())
        #author2author += scipy.sparse.
        author2author = author2author.nonzero()

        author_paper = self.paper_author[:,paper_idx]
        author_paper = author_paper[author_idx,:].nonzero()

        data = self.data.clone()
        data['paper'].x = data['paper'].x[paper_idx,:]
        data['paper']['mask'] = torch.tensor(len(paper_idx)).to(torch.long)
        data['author'].x = data['author'].x[author_idx,:]
        data['author']['mask'] = torch.tensor(len(author_idx)).to(torch.long)

        data['paper', 'subject', 'paper'].edge_index = torch.stack([torch.from_numpy(paper2paper[0]),
                                                                    torch.from_numpy(paper2paper[1])], dim=0)
        data['paper', 'subject', 'paper']['mask'] = torch.tensor(paper2paper[0].shape[0]).to(torch.long)

        data['author', 'Institute', 'author'].edge_index = torch.stack([torch.from_numpy(author2author[0]),
                                                                        torch.from_numpy(author2author[1])], dim=0)
        data['author', 'Institute', 'author']['mask'] = torch.tensor(author2author[0].shape[0]).to(torch.long)

        data['paper','from','author'].edge_index = torch.stack([torch.from_numpy(author_paper[1]),
                                                                torch.from_numpy(author_paper[0])], dim=0)
        data['paper','from','author']['mask'] = torch.tensor(author_paper[0].shape[0]).to(torch.long)

        data['author','write','paper'].edge_index = torch.stack([torch.from_numpy(author_paper[0]),
                                                                 torch.from_numpy(author_paper[1])], dim=0)
        data['author','write','paper']['mask'] = torch.tensor(author_paper[0].shape[0]).to(torch.long)

        data['paper'].y = torch.from_numpy(label.reshape([1,-1])).float()

        return data

if __name__=="__main__":
    import random

    seed = 10
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = OgbnMagDataset('train')

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    paper_dim = train_dataset.paper_dim
    author_dim = train_dataset.author_dim
    label_dim = train_dataset.y_dim

    for index, data in enumerate(train_dataloader):
        print("index ",index)  # ,end="  "



import os
import numpy as np
import scipy.sparse
from torch_geometric.data import HeteroData,DataLoader,Dataset
from scipy.io import loadmat,savemat
from  scipy.sparse import coo_matrix
import torch
import torch_geometric.transforms as T

data_dir = r'../exp/data_for_show/'
data_path = data_dir + r'/ogbn_mag_graph.mat'

def label_to_vector(index,len = 3):
    vec = np.zeros([1,len])
    vec[0,index]=1.0
    return vec

class ShowDataset(Dataset):
    def __init__(self, mode='train',
                 k = [16,8],
                 k1 = [16,8],
                 root=data_dir,
                 data_p = data_path,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,generate = False):
        self.generate = generate
        self.root = root

        self.paper_dim = 40 #self.Paper_Features.shape[1]
        self.author_dim = 40 #self.Author_Features.shape[1]
        self.y_dim = 3 #self.Paper_Label.shape[1]

        self.k = len(k)
        self.k_list = k
        self.k1_list = k1

        if mode in ['train','test','val']:
            self.mode = mode
        else:
            raise ValueError("mode only support  train test val")

        if not generate:
            lists = torch.load(os.path.join(root, "list_for_item.pt"))
            self.train_idx = lists[0]
            self.test_idx = lists[1]
            self.val_idx = lists[2]
        else:
            # datas = loadmat(data_p)
            label = np.load(data_dir+'select_paper_label.npy')
            self.Paper_Label = label

            paper_features_path = r"../exp/data_for_show/select_node_feature.npy"
            paper_features = np.load(paper_features_path, allow_pickle=True)
            self.paper_feature = paper_features

            author_features_path = r"../exp/data_for_show/select_author_feature.npy"
            author_features = np.load(author_features_path, allow_pickle=True)
            self.author_feature = author_features #datas['AT']

            paper_cite_paper_path = r"../exp/data_for_show/paperid_cite_paperid.npy"
            paper_cite_paper = np.load(paper_cite_paper_path, allow_pickle=True)
            self.paper_cite_paper = scipy.sparse.csr_matrix(paper_cite_paper) #datas['PCP']
            self.paper_degrees = self.paper_cite_paper.sum(axis=1)

            paper_author_path = r"../exp/data_for_show/select_paperid_authorid.npy"
            paper_author = np.load(paper_author_path, allow_pickle=True)
            self.paper_author = scipy.sparse.csr_matrix(paper_author) #datas['PA']
            self.PA_paper_degrees = self.paper_author.sum(axis=1)
            self.PA_author_degrees = self.paper_author.sum(axis=0)

            # paper subject
            # self.paper_subject = datas['PS']
            author_org_author_path = r"../exp/data_for_show/select_author_org_author.npy"
            author_org_author = np.load(author_org_author_path, allow_pickle=True)
            self.author_institution = scipy.sparse.csr_matrix(author_org_author) #datas['AFA']
            self.AF_author_degrees = self.author_institution.sum(axis=1)
            self.AF_institution_degrees = self.author_institution.sum(axis=0)

            splits_path = r"../exp/data_for_show/select_paper_split.npy"
            splits = np.load(splits_path, allow_pickle=True).item()
            self.train_idx = splits['train']
            self.val_idx = splits['test']
            self.test_idx = splits['test']

            data = HeteroData()
            data['paper'].x = torch.from_numpy(self.paper_feature).float()
            # data['paper'].y = self.Paper_Label
            data['author'].x = torch.from_numpy(self.author_feature).float()

            self.data = data
            # self.data = T.NormalizeFeatures()(self.data)
            # self.transform_0 = T.ToUndirected()
            # self.transform_1 = T.AddSelfLoops()

            self.domain_cross_sample = 16

        super(ShowDataset, self).__init__(root, transform, pre_transform, pre_filter)

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

        lists = [self.train_idx, self.test_idx, self.val_idx]
        #label_lists = [self.train_label, self.test_label, self.val_label]
        save_names = ["train", "test", "val"]
        torch.save(lists, os.path.join(self.root, "list_for_item.pt"))

        for ilist in range(len(lists)):
            list_for_iter = lists[ilist]
            label_for_iter =self.Paper_Label #label_lists[i]
            name_for_save = save_names[ilist]
            for item in range(len(list_for_iter)):
                id = item % len(list_for_iter)
                idx = list_for_iter[id]
                label = label_for_iter[id]
                label = label_to_vector(label)

                list_node_index = [idx]
                select = [idx]
                for i in range(self.k):
                    select_list = []
                    for ii in select:
                        indexs = self.paper_cite_paper[:, ii].nonzero()
                        select_tmp = []
                        for j in range(len(indexs[0])):
                            row = indexs[0][j]
                            col = indexs[1][j]
                            select_tmp.append(row)

                        indexs = self.paper_cite_paper[ii,:].nonzero()
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
                        args = degreelist[-self.k_list[i]:, 0].A[:, 0].tolist()
                        select_tmp = [select_tmp[i] for i in args]
                        # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                        select_list.extend(select_tmp)
                    select_list = list(set(select_list) - set(list_node_index))
                    select = select_list
                    list_node_index.extend(select_list)

                paper_idx = list_node_index
                index = np.argwhere(np.array(paper_idx) == idx)
                paper2paper = self.paper_cite_paper[paper_idx, :][:, paper_idx].nonzero()

                author_vector = self.paper_author[idx].nonzero()
                idx_lsit = author_vector[1].tolist()
                if len(idx_lsit) < self.domain_cross_sample:
                    select_idx_tmp = idx_lsit
                else:
                    degrees = self.PA_author_degrees[0, idx_lsit]
                    degreelist = np.argsort(degrees[:, 0], axis=0)
                    # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                    args = degreelist[-self.domain_cross_sample:, 0].A[:, 0].tolist()
                    select_idx_tmp = [idx_lsit[i] for i in args]
                    # select_idx_tmp = np.random.choice(idx_lsit, 5, replace=False).tolist()

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

                author_idx = list_node_index
                author_institution_sub = self.author_institution[author_idx, :]
                author2author = author_institution_sub.dot(author_institution_sub.transpose())
                # author2author += scipy.sparse.
                author2author = author2author.nonzero()

                author_paper = self.paper_author[ paper_idx,:]
                author_paper = author_paper[:,author_idx].T.nonzero()

                data = self.data.clone()
                data['paper'].x = data['paper'].x[paper_idx, :]
                data['paper']['mask'] = torch.tensor(index[0, 0]).to(torch.long)
                data['author'].x = data['author'].x[author_idx, :]
                data['author']['mask'] = torch.tensor(len(author_idx)).to(torch.long)

                data['paper', 'subject', 'paper'].edge_index = torch.stack([torch.from_numpy(paper2paper[0]),
                                                                            torch.from_numpy(paper2paper[1])], dim=0)
                data['paper', 'subject', 'paper']['mask'] = torch.tensor(paper2paper[0].shape[0]).to(torch.long)

                data['author', 'Institute', 'author'].edge_index = torch.stack([torch.from_numpy(author2author[0]),
                                                                                torch.from_numpy(author2author[1])],
                                                                               dim=0)
                data['author', 'Institute', 'author']['mask'] = torch.tensor(author2author[0].shape[0]).to(torch.long)

                data['paper', 'from', 'author'].edge_index = torch.stack([torch.from_numpy(author_paper[1]),
                                                                          torch.from_numpy(author_paper[0])], dim=0)
                data['paper', 'from', 'author']['mask'] = torch.tensor(author_paper[0].shape[0]).to(torch.long)

                data['author', 'write', 'paper'].edge_index = torch.stack([torch.from_numpy(author_paper[0]),
                                                                           torch.from_numpy(author_paper[1])], dim=0)
                data['author', 'write', 'paper']['mask'] = torch.tensor(author_paper[0].shape[0]).to(torch.long)

                data['paper'].y = torch.from_numpy(label.reshape([1, -1])).float()
                print(name_for_save + f"_data_{item}.pt")
                torch.save(data, os.path.join(self.processed_dir, name_for_save + f"_data_{item}.pt"))


    def len(self):
        if self.mode == 'train':
            return len(self.train_idx)
        elif self.mode == 'test':
            return len(self.test_idx)
        elif self.mode == 'val':
            return len(self.val_idx)

    def get(self, item):

        if self.mode == 'train':
            id = item % len(self.train_idx)
            data = torch.load(os.path.join(self.processed_dir,"train"+f"_data_{id}.pt"))
        elif self.mode == 'test':
            id = item % len(self.test_idx)
            data = torch.load(os.path.join(self.processed_dir,"test"+f"_data_{id}.pt"))
        elif self.mode == 'val':
            id = item % len(self.val_idx)
            data = torch.load(os.path.join(self.processed_dir,"val"+f"_data_{id}.pt"))

        return data

data_dirm = r'../exp/data_for_showm/'
class ShowDatasetM(Dataset):
    def __init__(self, mode='train',
                 k = [16,8],
                 k1 = [16,8],
                 root=data_dirm,
                 data_p = data_path,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,generate = False):
        self.generate = generate
        self.root = root

        self.paper_dim = 40 #self.Paper_Features.shape[1]
        self.author_dim = 40 #self.Author_Features.shape[1]
        self.y_dim = 3 #self.Paper_Label.shape[1]

        self.k = len(k)
        self.k_list = k
        self.k1_list = k1

        if mode in ['train','test','val']:
            self.mode = mode
        else:
            raise ValueError("mode only support  train test val")

        if not generate:
            lists = torch.load(os.path.join(root, "list_for_item.pt"))
            self.train_idx = lists[0]
            self.test_idx = lists[1]
            self.val_idx = lists[2]
        else:
            # datas = loadmat(data_p)
            label = np.load(data_dir+'select_paper_label.npy')
            self.Paper_Label = label

            paper_features_path = r"../exp/data_for_show/select_node_feature.npy"
            paper_features = np.load(paper_features_path, allow_pickle=True)
            self.paper_feature = paper_features

            author_features_path = r"../exp/data_for_show/select_author_feature.npy"
            author_features = np.load(author_features_path, allow_pickle=True)
            self.author_feature = author_features #datas['AT']

            paper_cite_paper_path = r"../exp/data_for_show/paperid_cite_paperid.npy"
            paper_cite_paper = np.load(paper_cite_paper_path, allow_pickle=True)
            self.paper_cite_paper = scipy.sparse.csr_matrix(paper_cite_paper) #datas['PCP']
            self.paper_degrees = self.paper_cite_paper.sum(axis=1)

            paper_author_path = r"../exp/data_for_show/select_paperid_authorid.npy"
            paper_author = np.load(paper_author_path, allow_pickle=True)
            self.paper_author = scipy.sparse.csr_matrix(paper_author) #datas['PA']
            self.PA_paper_degrees = self.paper_author.sum(axis=1)
            self.PA_author_degrees = self.paper_author.sum(axis=0)

            # paper subject
            # self.paper_subject = datas['PS']
            author_org_author_path = r"../exp/data_for_show/select_author_org_author.npy"
            author_org_author = np.load(author_org_author_path, allow_pickle=True)
            self.author_institution = scipy.sparse.csr_matrix(author_org_author) #datas['AFA']
            self.AF_author_degrees = self.author_institution.sum(axis=1)
            self.AF_institution_degrees = self.author_institution.sum(axis=0)

            splits_path = r"../exp/data_for_show/select_paper_split.npy"
            splits = np.load(splits_path, allow_pickle=True).item()
            self.train_idx = splits['train']
            self.val_idx = splits['test']
            self.test_idx = splits['test']

            paper_titles_path = r'../exp/data_for_show/select_paper_name.npy'
            paper_venue_path = r'../exp/data_for_show/select_paper_venue.npy'
            paper_year_path = r'../exp/data_for_show/select_paper_year.npy'

            self.paper_titles = np.load(paper_titles_path, allow_pickle=True)
            self.paper_venue = np.load(paper_venue_path, allow_pickle=True)
            self.paper_year = np.load(paper_year_path, allow_pickle=True)

            author_names_path = r'../exp/data_for_show/select_author_name.npy'
            author_org_path = r'../exp/data_for_show/select_author_org.npy'

            self.author_names = np.load(author_names_path, allow_pickle=True)
            self.author_org = np.load(author_org_path, allow_pickle=True)

            data = HeteroData()
            data['paper'].x = torch.from_numpy(self.paper_feature).float()
            # data['paper'].y = self.Paper_Label
            data['author'].x = torch.from_numpy(self.author_feature).float()

            self.data = data
            # self.data = T.NormalizeFeatures()(self.data)
            # self.transform_0 = T.ToUndirected()
            # self.transform_1 = T.AddSelfLoops()

            self.domain_cross_sample = 16

        super(ShowDatasetM, self).__init__(root, transform, pre_transform, pre_filter)

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

        lists = [self.train_idx, self.test_idx, self.val_idx]
        #label_lists = [self.train_label, self.test_label, self.val_label]
        save_names = ["train", "test", "val"]
        torch.save(lists, os.path.join(self.root, "list_for_item.pt"))

        for ilist in range(len(lists)):
            list_for_iter = lists[ilist]
            label_for_iter =self.Paper_Label #label_lists[i]
            name_for_save = save_names[ilist]
            for item in range(len(list_for_iter)):
                id = item % len(list_for_iter)
                idx = list_for_iter[id]
                label = label_for_iter[id]
                label = label_to_vector(label)

                list_node_index = [idx]
                select = [idx]
                for i in range(self.k):
                    select_list = []
                    for ii in select:
                        indexs = self.paper_cite_paper[:, ii].nonzero()
                        select_tmp = []
                        for j in range(len(indexs[0])):
                            row = indexs[0][j]
                            col = indexs[1][j]
                            select_tmp.append(row)

                        indexs = self.paper_cite_paper[ii,:].nonzero()
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
                        args = degreelist[-self.k_list[i]:, 0].A[:, 0].tolist()
                        select_tmp = [select_tmp[i] for i in args]
                        # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                        select_list.extend(select_tmp)
                    select_list = list(set(select_list) - set(list_node_index))
                    select = select_list
                    list_node_index.extend(select_list)

                paper_idx = list_node_index
                paper_name_list = self.paper_titles[paper_idx]
                paper_venue_list = self.paper_venue[paper_idx]
                paper_year_list = self.paper_year[paper_idx]
                index = np.argwhere(np.array(paper_idx) == idx)
                paper2paper = self.paper_cite_paper[paper_idx, :][:, paper_idx].nonzero()

                author_vector = self.paper_author[idx].nonzero()
                idx_lsit = author_vector[1].tolist()
                if len(idx_lsit) < self.domain_cross_sample:
                    select_idx_tmp = idx_lsit
                else:
                    degrees = self.PA_author_degrees[0, idx_lsit]
                    degreelist = np.argsort(degrees[:, 0], axis=0)
                    # select_tmp = np.random.choice(select_tmp, self.k_list[i], replace=False)
                    args = degreelist[-self.domain_cross_sample:, 0].A[:, 0].tolist()
                    select_idx_tmp = [idx_lsit[i] for i in args]
                    # select_idx_tmp = np.random.choice(idx_lsit, 5, replace=False).tolist()

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

                author_idx = list_node_index

                author_name_list = self.author_names[author_idx]
                author_org_list = self.author_org[author_idx]

                author_institution_sub = self.author_institution[author_idx, :]
                author2author = author_institution_sub.dot(author_institution_sub.transpose())
                # author2author += scipy.sparse.
                author2author = author2author.nonzero()

                author_paper = self.paper_author[ paper_idx,:]
                author_paper = author_paper[:,author_idx].T.nonzero()

                data = self.data.clone()
                data['paper'].x = data['paper'].x[paper_idx, :]
                data['paper']['mask'] = torch.tensor(index[0, 0]).to(torch.long)
                data['author'].x = data['author'].x[author_idx, :]
                data['author']['mask'] = torch.tensor(len(author_idx)).to(torch.long)

                data['paper', 'subject', 'paper'].edge_index = torch.stack([torch.from_numpy(paper2paper[0]),
                                                                            torch.from_numpy(paper2paper[1])], dim=0)
                data['paper', 'subject', 'paper']['mask'] = torch.tensor(paper2paper[0].shape[0]).to(torch.long)

                data['author', 'Institute', 'author'].edge_index = torch.stack([torch.from_numpy(author2author[0]),
                                                                                torch.from_numpy(author2author[1])],
                                                                               dim=0)
                data['author', 'Institute', 'author']['mask'] = torch.tensor(author2author[0].shape[0]).to(torch.long)

                data['paper', 'from', 'author'].edge_index = torch.stack([torch.from_numpy(author_paper[1]),
                                                                          torch.from_numpy(author_paper[0])], dim=0)
                data['paper', 'from', 'author']['mask'] = torch.tensor(author_paper[0].shape[0]).to(torch.long)

                data['author', 'write', 'paper'].edge_index = torch.stack([torch.from_numpy(author_paper[0]),
                                                                           torch.from_numpy(author_paper[1])], dim=0)
                data['author', 'write', 'paper']['mask'] = torch.tensor(author_paper[0].shape[0]).to(torch.long)

                data['paper'].y = torch.from_numpy(label.reshape([1, -1])).float()
                data['paper_title'] = paper_name_list
                data['paper_venue'] = paper_venue_list
                data['paper_year'] = paper_year_list

                data['author_name'] = author_name_list
                data['author_org'] = author_org_list
                print(name_for_save + f"_data_{item}.pt")
                torch.save(data, os.path.join(self.processed_dir, name_for_save + f"_data_{item}.pt"))


    def len(self):
        if self.mode == 'train':
            return len(self.train_idx)
        elif self.mode == 'test':
            return len(self.test_idx)
        elif self.mode == 'val':
            return len(self.val_idx)

    def get(self, item):

        if self.mode == 'train':
            id = item % len(self.train_idx)
            data = torch.load(os.path.join(self.processed_dir,"train"+f"_data_{id}.pt"))
        elif self.mode == 'test':
            id = item % len(self.test_idx)
            data = torch.load(os.path.join(self.processed_dir,"test"+f"_data_{id}.pt"))
        elif self.mode == 'val':
            id = item % len(self.val_idx)
            data = torch.load(os.path.join(self.processed_dir,"val"+f"_data_{id}.pt"))

        return data


if __name__=="__main__":
    import random

    seed = 10
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = ShowDatasetM('train',generate=True)

    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    #
    # paper_dim = train_dataset.paper_dim
    # author_dim = train_dataset.author_dim
    # label_dim = train_dataset.y_dim
    #
    # for index, data in enumerate(train_dataloader):
    #     print("index ",index)  # ,end="  "



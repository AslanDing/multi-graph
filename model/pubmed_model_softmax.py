import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from model.basic_layers import GNNNodeEmbed,EdgePred
import numpy as np

from model.layers_utils import gumbel_softmax

class MultiGraph(nn.Module):
    def __init__(self,x_dim=200,y_dim=8):
        super(MultiGraph, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.aim_key = 'domain1'
        self.node_keys = ['domain0','domain1','domain2','domain3','domain4']
        self.edge_keys = [('domain0', 'to', 'domain0'),
                          ('domain1', 'to', 'domain1'),
                          ('domain2', 'to', 'domain2'),
                          ('domain3', 'to', 'domain3'),

                          ('domain1', 'to', 'domain0'),
                          ('domain0', 'to', 'domain1'),
                          ('domain1', 'to', 'domain2'),
                          ('domain2', 'to', 'domain1'),
                          ('domain1', 'to', 'domain3'),
                          ('domain3', 'to', 'domain1')]


        self.multi_gnn_embedding = GNNNodeEmbed(x_dim,self.edge_keys,layer=2)

        self.edge_pred = EdgePred(self.node_keys,self.edge_keys[:4],self.multi_gnn_embedding.hidden_size)

        self.classify = nn.Linear(self.multi_gnn_embedding.hidden_size, y_dim)

    def forward(self,x, edge_index,batch,index):
        if self.training:
            x_embedding = self.multi_gnn_embedding(x,edge_index,single=True)
            edge_pred = self.edge_pred(x_embedding,edge_index)

            new_edge_indexs={}
            w_weight={}
            for key in edge_index.keys():
                if key[0]==key[2]:
                    shape = x_embedding[key[0]].shape # size=(shape[0],shape[0])
                    pred = edge_pred[key]
                    idx = edge_index[key]
                    if shape[0] <self.k:
                        new_edge_indexs[key]=idx
                        weight = torch.ones_like(pred[:,0])
                        w_weight[key] = weight
                        continue
                    spare_tensor = torch.sparse_coo_tensor(idx,pred[:,0],size=[shape[0],shape[0]],requires_grad=True).to_dense()
                    mask = torch.where(spare_tensor>0,torch.ones_like(spare_tensor),torch.zeros_like(spare_tensor))
                    spare_tensor = gumbel_softmax(spare_tensor,mask,self.k, 1, hard=True)
                    weight = spare_tensor[idx[0],idx[1]]
                    w_weight[key] = weight
                    new_edge = torch.nonzero(spare_tensor).T
                    new_edge_indexs[key] = new_edge
                else:
                    new_edge_indexs[key] = edge_index[key]


            multi_embedding = self.multi_gnn_embedding(x, edge_index,single=False,edge_attn=w_weight)
            multi_embedding_pool = multi_embedding[self.aim_key][index, :].reshape(1, -1)
            # multi_embedding_pool =global_add_pool(multi_embedding[self.aim_key],
            #                                       batch[self.aim_key])
            y_hat = self.classify(multi_embedding_pool)


            sub_x_embedding = self.multi_gnn_embedding(x,edge_index,single=True,edge_attn=w_weight)
            node_embedding_sub = {}
            for key in sub_x_embedding.keys():
                if sub_x_embedding[key].shape[0]<=1:
                    _embedding = sub_x_embedding[key]
                else:
                    _embedding = global_add_pool(sub_x_embedding[key], batch[key])
                node_embedding_sub[key] = _embedding

            src_multi_embedding = self.multi_gnn_embedding(x, edge_index,single=False)
            multi_src_embedding_pool = src_multi_embedding[self.aim_key][index, :].reshape(1, -1)
            # multi_src_embedding_pool = global_add_pool(src_multi_embedding[self.aim_key],
            #                                        batch[self.aim_key])
            return y_hat.softmax(dim=1), w_weight,\
                   multi_embedding_pool, multi_src_embedding_pool,\
                   node_embedding_sub
        else:
            x_embedding = self.multi_gnn_embedding(x, edge_index, single=True)
            edge_pred = self.edge_pred(x_embedding, edge_index)

            new_edge_indexs = {}
            w_weight = {}
            for key in edge_index.keys():
                if key[0] == key[2]:
                    shape = x_embedding[key[0]].shape  # size=(shape[0],shape[0])
                    pred = edge_pred[key]
                    idx = edge_index[key]
                    if shape[0] < self.k:
                        new_edge_indexs[key] = idx
                        weight = torch.ones_like(pred[:, 0])
                        w_weight[key] = weight
                        continue
                    spare_tensor = torch.sparse_coo_tensor(idx, pred[:, 0], size=[shape[0], shape[0]],
                                                           requires_grad=True).to_dense()
                    mask = torch.where(spare_tensor > 0, torch.ones_like(spare_tensor), torch.zeros_like(spare_tensor))
                    spare_tensor = gumbel_softmax(spare_tensor, mask, self.k, 1, hard=True)
                    weight = spare_tensor[idx[0], idx[1]]
                    w_weight[key] = weight
                    new_edge = torch.nonzero(spare_tensor).T
                    new_edge_indexs[key] = new_edge
                else:
                    new_edge_indexs[key] = edge_index[key]

            multi_embedding = self.multi_gnn_embedding(x, edge_index,single=False,edge_attn=w_weight)
            multi_embedding_pool = multi_embedding[self.aim_key][index, :].reshape(1, -1)
            # multi_embedding_pool = global_add_pool(multi_embedding[self.aim_key],
            #                                        batch[self.aim_key])
            y_hat = self.classify(multi_embedding_pool)
            return y_hat.softmax(dim=1),new_edge_indexs

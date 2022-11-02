import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from model.basic_layers import GNNNodeEmbed,EdgePred

class MultiGraph(nn.Module):
    def __init__(self,x_dim=1902,y_dim=3):
        super(MultiGraph, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.aim_key = 'paper'
        self.node_keys = ['paper','author']
        self.edge_keys = [('paper', 'subject', 'paper'),
                          ('author', 'Institute', 'author'),
                          ('paper','from','author'),
                          ('author','write','paper')]

        self.multi_gnn_embedding = GNNNodeEmbed(x_dim,self.edge_keys,layer=2)
        self.edge_pred = EdgePred(self.node_keys,self.edge_keys[:2],self.multi_gnn_embedding.hidden_size)

        self.classify = nn.Linear(self.multi_gnn_embedding.hidden_size, y_dim)

    def forward(self,x, edge_index,batch,index):
        if self.training:
            x_embedding = self.multi_gnn_embedding(x,edge_index,single=True)
            edge_pred = self.edge_pred(x_embedding,edge_index)

            z_dict = {}
            w_dict = {}
            # w_ones_dict = {}
            for key in edge_pred.keys():
                z = F.gumbel_softmax(edge_pred[key], 1, hard=True)
                z_dict[key] = z
                w = z[:, 1].squeeze()  # edge attention
                w_dict[key] = w
                # w_ones_dict[key] = torch.ones_like(w)

            #w_dict =  w_dict_list[0]
            multi_embedding = self.multi_gnn_embedding(x, edge_index,edge_attn=w_dict)
            multi_embedding_pool = multi_embedding[self.aim_key][index, :].reshape(1, -1)
            y_hat = self.classify(multi_embedding_pool)


            sub_x_embedding = self.multi_gnn_embedding(x,edge_index,edge_attn=w_dict,single=True)
            node_embedding_sub = {}
            for key in sub_x_embedding.keys():
                if sub_x_embedding[key].shape[0] <= 1:
                    _embedding = sub_x_embedding[key]
                else:
                    _embedding = global_add_pool(sub_x_embedding[key], batch[key])
                node_embedding_sub[key] = _embedding

            # multi_src_embedding_pool ={}
            src_multi_embedding = self.multi_gnn_embedding(x, edge_index,single=False)
            multi_src_embedding_pool = src_multi_embedding[self.aim_key][index, :].reshape(1, -1)

            return y_hat.softmax(dim=1), w_dict,\
                   multi_embedding_pool, multi_src_embedding_pool,\
                   node_embedding_sub
        else:
            x_embedding = self.multi_gnn_embedding(x, edge_index,single=True)
            edge_pred = self.edge_pred(x_embedding, edge_index)

            z_dict = {}
            w_dict = {}
            # w_ones_dict = {}
            for key in edge_pred.keys():
                z = F.gumbel_softmax(edge_pred[key], 1, hard=True)
                z_dict[key] = z
                w = z[:, 1].squeeze()  # edge attention
                w_dict[key] = w
            #     w_ones_dict[key] = torch.ones_like(w)

            multi_embedding = self.multi_gnn_embedding(x, edge_index,edge_attn=w_dict,single=False)
            multi_embedding_pool = multi_embedding[self.aim_key][index, :].reshape(1, -1)
            y_hat = self.classify(multi_embedding_pool)
            return y_hat.softmax(dim=1),w_dict

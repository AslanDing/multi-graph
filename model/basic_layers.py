import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GINEConv as BaseGINEConv, GINConv as BaseGINConv,HeteroConv

from torch_sparse import SparseTensor, fill_diag, matmul, mul
from typing import Union, Optional, List, Dict
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size, PairTensor

from torch_geometric.nn import HANConv
from model.gcn import GCNConv

def MLP(in_channels: int, out_channels: int):
    return nn.Sequential(
        Linear(in_channels, out_channels),
        nn.ReLU(inplace=True),
        Linear(out_channels, out_channels),
        #nn.LeakyReLU(0.2),
    )

class GINConv(BaseGINConv):
    def forward(self, x: Union[Tensor, OptPairTensor],
                      edge_index: Adj,
                      edge_attr: OptTensor = None,
                      edge_atten: OptTensor = None,
                      size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_atten=edge_atten, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor,
                      edge_atten: OptTensor = None) -> Tensor:
        if edge_atten is not None:
            return x_j * edge_atten.view((x_j.shape[0],1))
        else:
            return x_j

class GNNNodeEmbed(nn.Module):
    def __init__(self,x_dim,edge_dict_list,layer=2):
        super().__init__()
        self.n_layers = layer
        self.dropout_p = 0.6
        hidden_size = 256
        self.hidden_size = hidden_size
        self.node_encoder = Linear(x_dim, hidden_size)

        self.convs = nn.ModuleList()
        self.relu = nn.ELU()
        self.keys = edge_dict_list

        for ii in range(self.n_layers):
            list_heterconv = {}
            for keys in edge_dict_list:
                node0 = keys[0]
                node2 = keys[2]
                if node0 == node2:
                    list_heterconv[keys] = GINConv(MLP(hidden_size, hidden_size))
                else:
                    list_heterconv[keys] = GINConv(MLP(hidden_size, hidden_size))

            self.convs.append(HeteroConv(list_heterconv))

    def forward(self, x, edge_index,edge_attn=None,single=True):
        new_x = {}
        for key in x.keys():
            new_x[key] = self.node_encoder(x[key])

        new_edge_index = {}
        for key in self.keys:
            if single and key[0]!=key[2]:
                continue
            else:
                new_edge_index[key] = edge_index[key]

        for i in range(self.n_layers):
            if edge_attn == None:
                new_x = self.convs[i](new_x, new_edge_index)
            else:
                new_x = self.convs[i](new_x, new_edge_index, edge_attn)

            if i == self.n_layers -1 :
                continue
            for key in new_x.keys():
                new_x[key] = self.relu(new_x[key])
                new_x[key] = F.dropout(new_x[key], p=self.dropout_p, training=self.training)
        return new_x

class EdgePred(nn.Module):
    def __init__(self,node_keys,edge_keys,x_dim,y_dim=2):
        super(EdgePred, self).__init__()
        self.node_keys = node_keys
        self.edge_keys = edge_keys
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.edge_predict = nn.Sequential(Linear(self.x_dim *2, self.x_dim // 2),
                                          nn.ReLU(),
                                          Linear(self.x_dim// 2, 1),  # k = 2
                                          nn.LeakyReLU(0.2))

    def lift_node_att_to_edge_att(self,node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = torch.cat([src_lifted_att,dst_lifted_att],dim=-1)
        edge_att = self.edge_predict(edge_att)
        return edge_att

    def forward(self, x, edge_index):

        new_edge_pred = {}
        for index, key in enumerate(self.edge_keys):
            new_edge_pred[key] = self.lift_node_att_to_edge_att(x[self.node_keys[index]],
                                                                edge_index[key])
        return new_edge_pred
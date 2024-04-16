import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from basicmodel import (DenseAnchorDiffGraphConv, DenseAnchorGraphConv,
                        DenseDiffGraphConv, DenseGraphConv)
from utils import glorot


def dense_to_sparse(adj):
    edge_index = adj.nonzero().t()
    if edge_index.size(0) == 2:
        edge_attr = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_attr
    else:
        edge_attr = adj[edge_index[0], edge_index[1], edge_index[2]]
        row = edge_index[1] + adj.size(-2) * edge_index[0]
        col = edge_index[2] + adj.size(-1) * edge_index[0]
        return torch.stack([row, col], dim=0), edge_attr


def gumbel_softmax_sample(logits, tau, is_train=True, one_hot=False):
    if is_train:
        U = (-torch.empty_like(
            logits,
            memory_format=torch.legacy_contiguous_format).exponential_().log()
             )  # ~Gumbel(0,1)
        y = logits + U
        y = F.softmax(y / tau, dim=-1)
    else:
        y = logits
        y = F.softmax(y / tau, dim=-1)

    if one_hot:
        index = y.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format).scatter_(
                -1, index, 1.0)
        y = y_hard - y.detach() + y
    return y


class GumbelGCNConv(torch.nn.Module):

    def __init__(self, in_dim, out_dim, heads=4, K=5, tau=0.25):
        super().__init__()
        self.out_dim = out_dim
        self.heads = heads
        self.K = K
        self.tau = tau
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.lin_b = nn.Linear(out_dim, 1, bias=False)
        self.lin_c = nn.Linear(out_dim, 1, bias=False)

        self.gnn = DenseGraphConv(in_dim, out_dim)

    def reset_parameters(self):
        self.gnn.reset_parameters()
        glorot(self.lin.weight)
        glorot(self.lin_b.weight)
        glorot(self.lin_c.weight)  #

    def forward(self, x):
        q = self.lin(x)
        q = q.reshape(self.heads, -1, self.out_dim)  # h*n*d
        q = q / (torch.norm(q, p=2, keepdim=True) + 1e-10)

        logits = torch.matmul(q, q.transpose(-1, -2))
        b = torch.sigmoid(self.lin_b(q))
        c = torch.sigmoid(self.lin_c(q))
        logits = torch.exp((logits - b)**2 / (-0.1 * c))  # h*n*n

        logits = logits.repeat(self.K, 1, 1, 1)  # K*h*n*n

        if self.training:
            logits = gumbel_softmax_sample(logits, tau=self.tau, is_train=True)
        else:
            logits = gumbel_softmax_sample(logits,
                                           tau=self.tau,
                                           is_train=False)
        attention = torch.sum(logits, dim=0)  # h*n*n
        attention = torch.mean(attention, dim=0)  # n*n

        x = self.gnn(x, attention)
        return x


class GumbelDiffGCNConv(torch.nn.Module):

    def __init__(self, in_dim, out_dim, heads=4, K=5, tau=0.25):
        super().__init__()
        self.out_dim = out_dim
        self.heads = heads
        self.K = K
        self.tau = tau
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.lin_b = nn.Linear(out_dim, 1, bias=False)
        self.lin_c = nn.Linear(out_dim, 1, bias=False)

        self.gnn = DenseDiffGraphConv(in_dim, out_dim)

    def reset_parameters(self):
        self.gnn.reset_parameters()
        glorot(self.lin.weight)
        glorot(self.lin_b.weight)
        glorot(self.lin_c.weight)  #

    def forward(self, x):
        q = self.lin(x)  #
        q = q.reshape(self.heads, -1, self.out_dim)  # h*n*d
        q = q / (torch.norm(q, p=2, keepdim=True) + 1e-10)

        logits = torch.matmul(q, q.transpose(-1, -2))
        b = torch.sigmoid(self.lin_b(q))
        c = torch.sigmoid(self.lin_c(q))
        logits = torch.exp((logits + b)**2 / (-0.1 * c))  # h*n*n

        logits = logits.repeat(self.K, 1, 1, 1)  # K*h*n*n

        if self.training:
            logits = gumbel_softmax_sample(logits, tau=self.tau, is_train=True)
        else:
            logits = gumbel_softmax_sample(logits,
                                           tau=self.tau,
                                           is_train=False)
        attention = torch.sum(logits, dim=0)  # h*n*n
        attention = torch.mean(attention, dim=0)  # n*n

        x = self.gnn(x, attention)
        return x
    

class DualGumelGCNConv(nn.Module):

    def __init__(self, in_dim, out_dim, heads=4, K=5, tau=0.25):
        super().__init__()
        self.addgcnconv = GumbelGCNConv(in_dim, out_dim, heads, K, tau)
        self.diffgcnconv = GumbelDiffGCNConv(in_dim, out_dim, heads, K, tau)

    def reset_parameters(self):
        self.addgcnconv.reset_parameters()
        self.diffgcnconv.reset_parameters()

    def forward(self, x):
        x_1 = self.addgcnconv(x)
        x_2 = self.diffgcnconv(x)
        x = (x_1 + x_2) / 2.
        return x


class GumbelAnchorGCNConv(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 num_nodes,
                 anchor_nodes,
                 heads=4,
                 K=5,
                 tau=0.25):
        super().__init__()
        self.out_dim = out_dim
        self.heads = heads
        self.K = K
        self.tau = tau
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.lin_b = nn.Linear(out_dim, 1, bias=False)
        self.lin_c = nn.Linear(out_dim, 1, bias=False)

        self.E = Parameter(torch.Tensor(anchor_nodes, num_nodes))
        self.gnn = DenseAnchorGraphConv(in_dim, out_dim, anchor_nodes,
                                        num_nodes)

    def reset_parameters(self):
        self.gnn.reset_parameters()
        glorot(self.lin.weight)
        glorot(self.E)
        glorot(self.lin_b.weight)  #
        glorot(self.lin_c.weight)  #

    def forward(self, x):
        q = self.lin(x)
        q = q.reshape(self.heads, -1, self.out_dim)  # h*n*d
        anchor_q = torch.matmul(self.E, q)  # s*n h*n*d => h*s*d
        q = q / (torch.norm(q, p=2, keepdim=True) + 1e-10)
        anchor_q = anchor_q / (torch.norm(anchor_q, p=2, keepdim=True) + 1e-10)

        logits = torch.matmul(q, anchor_q.transpose(-1, -2))  # h*n*s
        b = torch.sigmoid(self.lin_b(q))  # h*n*1
        c = torch.sigmoid(self.lin_c(q))  # h*n*1
        logits = torch.exp((logits - b)**2 / (-0.1 * c))

        logits = logits.repeat(self.K, 1, 1, 1)  # K*h*n*s

        if self.training:
            logits = gumbel_softmax_sample(logits, tau=self.tau, is_train=True)
        else:
            logits = gumbel_softmax_sample(logits,
                                           tau=self.tau,
                                           is_train=False)
        attention = torch.sum(logits, dim=0)  # h*n*s
        attention = torch.mean(attention, dim=0)  # n*s

        x = self.gnn(x, attention)
        return x


class GumbelAnchorDiffGCNConv(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 num_nodes,
                 anchor_nodes,
                 heads=4,
                 K=5,
                 tau=0.25):
        super().__init__()
        self.out_dim = out_dim
        self.heads = heads
        self.K = K
        self.tau = tau
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.lin_b = nn.Linear(out_dim, 1, bias=False)
        self.lin_c = nn.Linear(out_dim, 1, bias=False)

        self.E = Parameter(torch.Tensor(anchor_nodes, num_nodes))
        self.gnn = DenseAnchorDiffGraphConv(in_dim, out_dim, anchor_nodes,
                                            num_nodes)
        # self.eps = torch.nn.Parameter(torch.Tensor(1, ))

    def reset_parameters(self):
        self.gnn.reset_parameters()
        glorot(self.lin.weight)
        glorot(self.E)
        glorot(self.lin_b.weight)  #
        glorot(self.lin_c.weight)  #
        # self.lin_c.weight.data.fill_(0.05437)

    def forward(self, x):
        q = self.lin(x)
        q = q.reshape(self.heads, -1, self.out_dim)  # h*n*d
        anchor_q = torch.matmul(self.E, q)  # s*n h*n*d => h*s*d
        q = q / (torch.norm(q, p=2, keepdim=True) + 1e-10)
        anchor_q = anchor_q / (torch.norm(anchor_q, p=2, keepdim=True) + 1e-10)

        logits = torch.matmul(q, anchor_q.transpose(-1, -2))  # h*n*s
        b = torch.sigmoid(self.lin_b(q))  # h*n*1
        c = torch.sigmoid(self.lin_c(q))  # h*n*1
        logits = torch.exp((logits + b)**2 / (-0.1 * c))

        logits = logits.repeat(self.K, 1, 1, 1)  # K*h*n*s

        if self.training:
            logits = gumbel_softmax_sample(logits, tau=self.tau, is_train=True)
        else:
            logits = gumbel_softmax_sample(logits,
                                           tau=self.tau,
                                           is_train=False)
        attention = torch.sum(logits, dim=0)  # h*n*s
        attention = torch.mean(attention, dim=0)  # n*s

        x = self.gnn(x, attention)
        return x


class DualGumbelAnchorGCNConv(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 num_nodes,
                 anchor_nodes,
                 heads=4,
                 K=5,
                 tau=0.25):
        super().__init__()
        self.addgcnconv = GumbelAnchorGCNConv(in_dim, out_dim, num_nodes,
                                              anchor_nodes, heads, K, tau)
        self.diffgcnconv = GumbelAnchorDiffGCNConv(in_dim, out_dim, num_nodes,
                                                   anchor_nodes, heads, K, tau)

    def reset_parameters(self):
        self.addgcnconv.reset_parameters()
        self.diffgcnconv.reset_parameters()

    def forward(self, x):
        x_1 = self.addgcnconv(x)
        x_2 = self.diffgcnconv(x)
        # out = torch.cat([x_1, x_2], dim=-1)
        out = (x_1 + x_2) / 2.
        return out

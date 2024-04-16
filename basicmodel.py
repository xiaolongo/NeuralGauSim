import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros


class DenseGraphConv(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.lin_rel = Linear(in_dim, out_dim)
        self.lin_root = Linear(in_dim, out_dim, bias=False)

    def reset_parameters(self):
        glorot(self.lin_rel.weight)
        glorot(self.lin_root.weight)
        zeros(self.lin_rel.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = out / adj.sum(dim=-1, keepdim=True).clamp_(min=1)

        out = self.lin_rel(out)
        out = out + self.lin_root(x)
        return out


class DenseDiffGraphConv(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.lin_rel = Linear(in_dim, out_dim)
        self.lin_root = Linear(in_dim, out_dim, bias=False)

    def reset_parameters(self):
        glorot(self.lin_rel.weight)
        glorot(self.lin_root.weight)
        zeros(self.lin_rel.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = out / adj.sum(dim=-1, keepdim=True).clamp_(min=1)

        out = self.lin_rel(out)
        out = out - self.lin_root(x)
        return out


class DenseAnchorGraphConv(torch.nn.Module):

    def __init__(self, in_dim, out_dim, anchor_nodes, num_nodes):
        super().__init__()

        self.lin_rel = Linear(in_dim, out_dim)
        self.lin_root = Linear(in_dim, out_dim)

        self.F = Parameter(torch.Tensor(anchor_nodes, num_nodes))

    def reset_parameters(self):
        glorot(self.lin_rel.weight)
        glorot(self.lin_root.weight)
        glorot(self.F)
        zeros(self.lin_rel.bias)

    def forward(self, x, adj):
        out = F.elu(torch.matmul(self.F, x))
        out = torch.matmul(adj, out)
        out = out / adj.sum(dim=-1, keepdim=True).clamp_(min=1)

        out = self.lin_rel(out)
        out = self.lin_root(x) + out
        return out


class DenseAnchorDiffGraphConv(torch.nn.Module):

    def __init__(self, in_dim, out_dim, anchor_nodes, num_nodes):
        super().__init__()

        self.lin_rel = Linear(in_dim, out_dim)
        self.lin_root = Linear(in_dim, out_dim)

        self.F = Parameter(torch.Tensor(anchor_nodes, num_nodes))

    def reset_parameters(self):
        glorot(self.lin_rel.weight)
        glorot(self.lin_root.weight)
        glorot(self.F)
        zeros(self.lin_rel.bias)

    def forward(self, x, adj):
        out = F.elu(torch.matmul(self.F, x))
        out = torch.matmul(adj, out)
        out = out / adj.sum(dim=-1, keepdim=True).clamp_(min=1)

        out = self.lin_rel(out)
        out = self.lin_root(x) - out
        return out

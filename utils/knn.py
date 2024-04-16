import torch
from torch_geometric.utils import dense_to_sparse


def cosine_similarity(x):
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    x_norm = x.div(x_norm + 5e-10)
    cos_adj = torch.mm(x_norm, x_norm.transpose(0, 1))
    return cos_adj


def build_knn_graph(x, k):
    '''generating knn graph from data x
    Args: 
        x: input data (n, m)
        k: number of nearst neighbors           
    returns:
        knn_edge_index, knn_edge_weight
    '''
    cos_adj = cosine_similarity(x)
    topk = min(k + 1, cos_adj.size(-1))
    knn_val, knn_ind = torch.topk(cos_adj, topk, dim=-1)
    weighted_adj = (torch.zeros_like(cos_adj)).scatter_(-1, knn_ind, knn_val)
    knn_edge_index, knn_edge_weight = dense_to_sparse(weighted_adj)
    return knn_edge_index, knn_edge_weight


def build_dilated_knn_graph(x, k1, k2):
    '''generating dilated knn graph from data x
    Args:
        x: input data (n, m)
        k1: number of nearst neighbors
        k2: number of dilations           
    returns:
        knn_edge_index, knn_edge_weight
    '''
    cos_adj = cosine_similarity(x)
    topk = min(k1 + 1, cos_adj.size(-1))
    knn_val, knn_ind = torch.topk(cos_adj, topk, dim=-1)
    knn_val = knn_val[:, k2:]  #
    knn_ind = knn_ind[:, k2:]  #
    weighted_adj = (torch.zeros_like(cos_adj)).scatter_(-1, knn_ind, knn_val)
    knn_edge_index, knn_edge_weight = dense_to_sparse(weighted_adj)
    return knn_edge_index, knn_edge_weight


def build_epsilon_graph(x, epsilon):
    '''generating epsilon knn graph from data x
    Args:
        x: input data (n, m)
        epsilon: cosine_simility > epsilon          
    returns:
        knn_edge_index, knn_edge_weight
    '''
    cos_adj = cosine_similarity(x)
    mask = (cos_adj > epsilon).detach().float()
    weighted_adj = cos_adj * mask
    knn_edge_index, knn_edge_weight = dense_to_sparse(weighted_adj)
    return knn_edge_index, knn_edge_weight


def build_dilated_epsilon_graph(x, epsilon1, epsilon2):
    '''generating dilated epsilon knn graph from data x
    Args:
        x: input data (n, m)
        epsilon1: cosine_simility > epsilon1
        epsilon2: cosine_simility < epsilon2          
    returns:
        knn_edge_index, knn_edge_weight
    '''
    cos_adj = cosine_similarity(x)
    mask1 = (cos_adj > epsilon1).detach().float()
    mask2 = (cos_adj < epsilon2).detach().float()
    weighted_adj = cos_adj * mask1 * mask2
    weighted_adj = (weighted_adj + weighted_adj.T) / 2.
    knn_edge_index, knn_edge_weight = dense_to_sparse(weighted_adj)
    return knn_edge_index, knn_edge_weight


def knn_fast(x, k, b=10000, device='cpu'):
    '''block based fast KNN search algorithm
    Args:
        x: input data
        k: number of nearst neighbors
        b: number of blocks
        device: tensor device ('cuda' or 'cpu')
    returns:
        knn_edge_index, knn_edge_weight
    '''
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    x = x.div(x_norm + 5e-10)
    index = 0
    edge_weight = torch.zeros(x.shape[0] * (k + 1)).to(device)
    rows = torch.zeros(x.shape[0] * (k + 1)).to(device)
    cols = torch.zeros(x.shape[0] * (k + 1)).to(device)
    while index < x.shape[0]:
        if (index + b) > (x.shape[0]):
            end = x.shape[0]
        else:
            end = index + b
        sub_tensor = x[index:index + b]
        similarities = torch.mm(sub_tensor, x.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        vals = vals[:, :].contiguous().detach()
        inds = inds[:, :].contiguous().detach()
        edge_weight[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(
            -1, 1).repeat(1, k + 1).view(-1)
        index += b
    rows, cols = rows.long(), cols.long()
    edge_index = torch.cat([rows.unsqueeze(0), cols.unsqueeze(0)], dim=0)
    return edge_index, edge_weight


def heat_knn_fast(x, k, b=10000, device='cpu'):
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    x = x.div(x_norm + 5e-10)
    index = 0
    edge_weight = torch.zeros(x.shape[0] * (k + 1)).to(device)
    rows = torch.zeros(x.shape[0] * (k + 1)).to(device)
    cols = torch.zeros(x.shape[0] * (k + 1)).to(device)
    while index < x.shape[0]:
        if (index + b) > (x.shape[0]):
            end = x.shape[0]
        else:
            end = index + b
        sub_tensor = x[index:index + b]
        similarities = torch.mm(sub_tensor, x.t())
        similarities = torch.exp(-similarities / 2)  #
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        vals = vals[:, :].contiguous().detach()
        inds = inds[:, :].contiguous().detach()
        edge_weight[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(
            -1, 1).repeat(1, k + 1).view(-1)
        index += b
    rows, cols = rows.long(), cols.long()
    edge_index = torch.cat([rows.unsqueeze(0), cols.unsqueeze(0)], dim=0)
    return edge_index, edge_weight

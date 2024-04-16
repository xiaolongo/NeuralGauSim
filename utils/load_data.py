import math
import os.path as osp
import pickle as pkl

import numpy as np
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from torch_geometric.data import Data
from torch_geometric.datasets import (Actor, Coauthor, Flickr, LINKXDataset,
                                      Planetoid, WebKB, WikipediaNetwork)
from torch_geometric.utils import to_undirected


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def rand_train_test_idx(data,
                        train_prop=.5,
                        val_prop=.25,
                        ignore_negative=True,
                        mask=True):
    label = data.y
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * val_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    val_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    if mask:
        data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
        data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
        data.test_mask = index_to_mask(test_idx, size=data.num_nodes)
    else:
        data.train_mask = train_idx
        data.val_mask = val_idx
        data.test_mask = test_idx

    return data


def class_rand_splits(data, label_num_per_class=20):
    label = data.y
    train_idx, non_train_idx = [], []
    idx = torch.arange(label.shape[0])
    class_list = label.squeeze().unique()
    valid_num, test_num = 500, 1000
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:label_num_per_class].tolist()
        non_train_idx += rand_idx[label_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    non_train_idx = torch.as_tensor(non_train_idx)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]
    val_idx, test_idx = non_train_idx[:valid_num], non_train_idx[
        valid_num:valid_num + test_num]

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)

    return data


def ogb_splits(dataset, data):
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)

    return data


def load_planetoid(dataset, split_type='random', mask=True):
    data_name = ['Cora', 'CiteSeer', 'PubMed']
    assert dataset in data_name
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..',
                    'Datasets', 'NodeData')
    transforms = T.Compose([T.NormalizeFeatures()])
    dataset = Planetoid(path, dataset, transform=transforms)
    data = dataset[0]
    if split_type == 'public':
        pass
    elif split_type == 'random':
        data = rand_train_test_idx(data,
                                   train_prop=0.6,
                                   val_prop=0.2,
                                   mask=mask)
    elif split_type == 'class':
        data = class_rand_splits(data, label_num_per_class=20)
    else:
        raise NotImplementedError
    return dataset, data


def load_actor(dataset, split_type='random', mask=True):
    data_name = ['Actor']
    assert dataset in data_name
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..',
                    'Datasets', 'NodeData', dataset)
    transforms = T.Compose([T.NormalizeFeatures()])
    dataset = Actor(path, transform=transforms)
    data = dataset[0]
    if split_type == 'random':
        data = rand_train_test_idx(data,
                                   train_prop=0.6,
                                   val_prop=0.2,
                                   mask=mask)
    elif split_type == 'class':
        data = class_rand_splits(data, label_num_per_class=20)
    else:
        raise NotImplementedError
    return dataset, data


def load_wikipedia(dataset, split_type='random', mask=True):
    data_name = ['Chameleon', 'Squirrel']
    assert dataset in data_name
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..',
                    'Datasets', 'NodeData')
    transforms = T.Compose([T.NormalizeFeatures()])
    dataset = WikipediaNetwork(path, dataset, transform=transforms)
    data = dataset[0]
    if split_type == 'random':
        data = rand_train_test_idx(data,
                                   train_prop=0.6,
                                   val_prop=0.2,
                                   mask=mask)
    elif split_type == 'class':
        data = class_rand_splits(data, label_num_per_class=20)
    else:
        raise NotImplementedError
    return dataset, data


def load_webkb(dataset, split_type='random', mask=True):
    data_name = ['Cornell', 'Texas', 'Wisconsin']
    assert dataset in data_name
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..',
                    'Datasets', 'NodeData')
    transforms = T.Compose([T.NormalizeFeatures()])
    dataset = WebKB(path, dataset, transform=transforms)
    data = dataset[0]
    if split_type == 'random':
        data = rand_train_test_idx(data,
                                   train_prop=0.6,
                                   val_prop=0.2,
                                   mask=mask)
    elif split_type == 'class':
        data = class_rand_splits(data, label_num_per_class=20)
    else:
        raise NotImplementedError
    return dataset, data


def load_20news(mask=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..',
                    'Datasets', 'TextData')
    data = pkl.load(open(path + '/20news/20news.pkl', 'rb'))

    vectorizer = CountVectorizer(stop_words='english', min_df=0.05)
    X_counts = vectorizer.fit_transform(data.data).toarray()
    transformer = TfidfTransformer(smooth_idf=True)
    features = transformer.fit_transform(X_counts).todense()
    features = torch.Tensor(features)
    labels = data.target
    labels = torch.LongTensor(labels)
    c = labels.max().item() + 1

    data = Data(x=features, y=labels, num_classes=c)
    data = rand_train_test_idx(data, train_prop=0.6, val_prop=0.2, mask=mask)
    return data, data


def load_mini_imagenet(mask=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..',
                    'Datasets', 'ImageData')
    data = pkl.load(open(path + '/mini_imagenet/mini_imagenet.pkl', 'rb'))
    x_train = data['x_train']
    x_val = data['x_val']
    x_test = data['x_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    features = torch.cat((x_train, x_val, x_test), dim=0)
    labels = np.concatenate((y_train, y_val, y_test))
    labels = torch.LongTensor(labels)
    c = labels.max().item() + 1

    data = Data(x=features, y=labels, num_classes=c)
    data = rand_train_test_idx(data, train_prop=0.6, val_prop=0.2, mask=mask)
    return data, data


def load_homo_dataset(name, mask=True):
    assert name in [
        'amazon-ratings', 'minesweeper', 'questions', 'roman-empire',
        'tolokers'
    ]
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..',
                    'Datasets', 'NodeData', f'{name.replace("-", "_")}',
                    f'{name.replace("-", "_")}.npz')
    data = np.load(path)
    features = torch.tensor(data['node_features'])
    # features = (features - features.min(dim=1, keepdim=True)[0]) / torch.sum(
    #     features, dim=1, keepdim=True)
    labels = torch.tensor(data['node_labels'])
    edges = torch.tensor(data['edges'])
    edges = to_undirected(edge_index=edges.transpose(-1, -2),
                          num_nodes=features.shape[0])
    c = len(labels.unique())

    data = Data(x=features, edge_index=edges, y=labels, num_classes=c)
    data = rand_train_test_idx(data, train_prop=0.6, val_prop=0.2, mask=mask)
    return data, data


def load_ogb_dataset(name, mask=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..',
                    'Datasets', 'OGBData')
    transforms = T.Compose([T.ToSparseTensor()])
    dataset = PygNodePropPredDataset(name=name,
                                     root=path,
                                     transform=transforms)
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    row, col, _ = data.adj_t.t().coo()
    data.edge_index = torch.stack([row, col], dim=0)
    data.y = data.y.squeeze(1)
    # data = ogb_splits(dataset, data)
    data = rand_train_test_idx(data, train_prop=0.6, val_prop=0.2, mask=mask)
    return dataset, data


def load_linkx_dataset(name, mask=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..',
                    'Datasets', 'NodeData')
    transforms = T.Compose([T.NormalizeFeatures()])
    dataset = LINKXDataset(root=path, name=name, transform=transforms)

    data = dataset[0]
    data = rand_train_test_idx(data, train_prop=0.6, val_prop=0.2, mask=mask)
    return dataset, data


def load_flickr_dataset(mask=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..',
                    'Datasets', 'NodeData', 'Flickr')
    transforms = T.Compose([T.NormalizeFeatures()])
    dataset = Flickr(root=path, transform=transforms)

    data = dataset[0]
    data = rand_train_test_idx(data, train_prop=0.6, val_prop=0.2, mask=mask)
    return dataset, data


def load_coauthor_dataset(name, mask=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..',
                    'Datasets', 'NodeData')
    transforms = T.Compose([T.NormalizeFeatures()])
    dataset = Coauthor(root=path, name=name, transform=transforms)
    data = dataset[0]
    data = rand_train_test_idx(data, train_prop=0.6, val_prop=0.2, mask=mask)
    return dataset, data


def load_data(name, split_type='random', mask=True):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset, data = load_planetoid(name, split_type=split_type, mask=mask)
    elif name in ['Actor']:
        dataset, data = load_actor(name, split_type=split_type, mask=mask)
    elif name in ['Chameleon', 'Squirrel']:
        dataset, data = load_wikipedia(name, split_type=split_type, mask=mask)
    elif name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset, data = load_webkb(name, split_type=split_type, mask=mask)
    elif name in ['20News']:
        dataset, data = load_20news(mask=mask)
    elif name in ['Mini']:
        dataset, data = load_mini_imagenet(mask=mask)
    elif name in ['ogbn-arxiv']:
        dataset, data = load_ogb_dataset(name, mask=mask)
    elif name in ['Flickr']:
        dataset, data = load_flickr_dataset(mask=mask)
    elif name in ['CS', 'Physics']:
        dataset, data = load_coauthor_dataset(name, mask=mask)
    elif name in [
            'penn94', 'reed98', 'amherst41', 'cornell5', 'johnshopkins55',
            'genius'
    ]:
        dataset, data = load_linkx_dataset(name, mask=mask)
    elif name in [
            'amazon-ratings', 'minesweeper', 'questions', 'roman-empire',
            'tolokers'
    ]:
        dataset, data = load_homo_dataset(name, mask=mask)
    else:
        raise NotImplementedError
    return dataset, data

import os.path as osp

import torch
from torch_geometric.datasets import TUDataset
from tudataset import TUDatasetExt
from torch_geometric.utils import degree
import torch_geometric.transforms as T
import torch.nn.functional as F


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

class OneHotDegree(object):
    r"""Adds the node degree as one hot encodings to the node features.

    Args:
        max_degree (int): Maximum degree.
        in_degree (bool, optional): If set to :obj:`True`, will compute the
            in-degree of nodes instead of the out-degree.
            (default: :obj:`False`)
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)
    """

    def __init__(self, max_degree, in_degree=False, cat=True):
        self.max_degree = max_degree
        self.in_degree = in_degree
        self.cat = cat

    def __call__(self, data):
        idx, x = data.edge_index[1 if self.in_degree else 0], data.x
        deg = degree(idx, data.num_nodes,
                dtype=torch.long).clamp(max=self.max_degree)
        deg = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.max_degree)



def get_dataset(name, sparse=True, cleaned=False, aug="dropN", aug_ratio=0.2):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = TUDataset(path, name, cleaned=cleaned, use_node_attr=True,
            use_edge_attr=True)
    dataset_aug = TUDatasetExt(path, name, use_node_attr=True, aug=aug,
            aug_ratio=aug_ratio)
    dataset.data.edge_attr = None
    dataset_aug.data.edge_attr = None

    if False: #dataset.data.x is not None:
        dataset.data.x[:, :dataset.num_node_attributes] -= \
               dataset.data.x[:, :dataset.num_node_attributes].mean(dim=0, keepdims=True)
        dataset.data.x[:, :dataset.num_node_attributes] /= \
                dataset.data.x[:, :dataset.num_node_attributes].std(dim=0, keepdims=True)
    if dataset.data.x is None:
        dataset.degree_feature = True
        dataset_aug.degree_feature = True
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = OneHotDegree(max_degree)
            dataset_aug.transform = OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)
            dataset_aug.transform = NormalizedDegree(mean, std)
    else:
        dataset.degree_feature = False
        dataset_aug.degree_feature = False

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset[torch.tensor(indices)]

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])

    return dataset, dataset_aug

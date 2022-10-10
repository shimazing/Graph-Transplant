import os.path as osp

import torch
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_dataset(name, sparse=True, cleaned=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    if name == 'CIFAR10':
        train_dataset = GNNBenchmarkDataset(path, name, split='train', use_node_attr=True,
                use_edge_attr=False, cleaned=cleaned)
        val_dataset = GNNBenchmarkDataset(path, name, split='val', use_node_attr=True,
                use_edge_attr=False, cleaned=cleaned)
        test_dataset = GNNBenchmarkDataset(path, name, split='test', use_node_attr=True,
                use_edge_attr=False, cleaned=cleaned)
        train_dataset.degree_feature = False
        val_dataset.degree_feature = False
        test_dataset.degree_feature = False
        train_dataset.data.edge_attr = None
        val_dataset.data.edge_attr = None
        test_dataset.data.edge_attr = None
        return [train_dataset, val_dataset, test_dataset]

    dataset = TUDataset(path, name, cleaned=cleaned, use_node_attr=True,
        use_edge_attr=True)
    dataset.data.edge_attr = None

    if False: #dataset.data.x is not None:
        dataset.data.x[:, :dataset.num_node_attributes] -= \
               dataset.data.x[:, :dataset.num_node_attributes].mean(dim=0, keepdims=True)
        dataset.data.x[:, :dataset.num_node_attributes] /= \
                dataset.data.x[:, :dataset.num_node_attributes].std(dim=0, keepdims=True)
    if dataset.data.x is None:
        dataset.degree_feature = True
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)
    else:
        dataset.degree_feature = False

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

    return dataset

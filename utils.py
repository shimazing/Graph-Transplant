import torch
from torch_scatter import scatter_add, scatter_max
from torch_geometric.utils import to_dense_batch, sort_edge_index,\
                                add_self_loops


def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target', ratio=1.,
                   batch=None):
    # saliency가 주어지면 그걸로 edge_weight만들고 normalize 해서 random walk에
    # 씀.
    # 여기 batch는 기존의 data.batch
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """
    num_nodes = num_nodes if num_nodes is not None else edge_index[0].max() + 1

    assert flow in ['source_to_target', 'target_to_source']
    assert torch.all(edge_index[0] != edge_index[1]), "should not contain self-loop"

    row, col = edge_index
    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)
    #focus = node_idx
    subsets = [node_idx]

    for _ in range(num_hops):
        # original
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        # row
        subsets.append(torch.unique(col[edge_mask]))
        if ratio < 1:
            new_selected = len(subsets[-1])
            subsets = subsets[:-1] + [subsets[-1][torch.randperm(new_selected)[:max(1, int(ratio *
                new_selected))].to(subsets[-1].device)]]
    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]
    node_mask.fill_(False)
    node_mask[subset] = True
    row, col = edge_index
    edge_mask = node_mask[row] & node_mask[col]
    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]
    return subset, edge_index, inv, edge_mask

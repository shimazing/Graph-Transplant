import time
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch, subgraph, sort_edge_index,\
        to_dense_adj, dense_to_sparse
from utils import k_hop_subgraph
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from old_subgraph_train_eval import k_fold
from torch_sparse import spspmm
from torch_sparse.tensor import SparseTensor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

saliency = None

import copy
def cross_validation_with_val_set(dataset, dataset_aug, model, ep_net, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, method, logger=None, k=1,
                                  ratio=0.5, manifold=False, edge_predict=True,
                                  edge_thrs=0.5, only_aug=False, smoothing=0.0,
                                  train_reduce=1):
    epochs = epochs * train_reduce
    val_accs, val_losses, accs, durations = [], [], [], []
    repeat = 3
    accum_steps = 1
    for _ in range(repeat):
      for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds, train_reduce))):
        dataset_ = copy.deepcopy(dataset)
        dataset_aug_ = copy.deepcopy(dataset_aug)
        if dataset.num_node_attributes > 0:
            n_attr = dataset.num_node_attributes
            mean = dataset.data.x[torch.cat((train_idx, val_idx))][:, :n_attr].mean(dim=0,
                    keepdims=True)
            std = dataset.data.x[torch.cat((train_idx, val_idx))][:, :n_attr].std(dim=0,
                    keepdims=True)
            dataset_.data.x[:, :n_attr] -= mean
            dataset_.data.x[:, :n_attr] /= std
            dataset_aug_.data.x[:, :n_attr] -= mean
            dataset_aug_.data.x[:, :n_attr] /= std


        train_dataset = dataset_[train_idx]
        train_aug_dataset = dataset_aug_[train_idx]
        test_dataset = dataset_[test_idx]
        val_dataset = dataset_[val_idx]


        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        ep_net.to(device)
        for m in ep_net:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        optimizer = Adam(list(model.parameters()) + list(ep_net.parameters()), lr=lr, weight_decay=weight_decay)
        iters = len(train_loader) / accum_steps
        if lr_decay_factor < 1:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=lr_decay_factor,
                                                             patience=int(1000 / iters),
                                                             verbose=False)

        t_start = time.perf_counter()
        stop_patience = 0
        best_acc = 0
        avg_class_prediction = None
        theta = None
        for epoch in tqdm(range(1, epochs + 1)):
            if stop_patience == int(1500/iters):
                train_loss = -1
                val_losses.append(100)
                val_accs.append(-1)
                accs.append(-1)
                continue
            train_loss = train(model, ep_net, optimizer, train_loader,
                    method,
                    accum_steps=accum_steps, k=k, ratio=ratio,
                    manifold=manifold, degree_feature=dataset.degree_feature,
                    edge_predict=edge_predict, edge_thrs=edge_thrs,
                    only_aug=only_aug, smoothing=smoothing,
                    avg_class_prediction=avg_class_prediction,
                    theta=theta,
                    )
            val_losses.append(eval_loss(model, val_loader))
            val_acc, avg_class_prediction, theta = eval_acc(model, val_loader)
            val_accs.append(val_acc)
            current_val_acc = val_accs[-1]
            accs.append(eval_acc(model, test_loader)[0])
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_acc': val_accs[-1],
                'val_loss': val_losses[-1],
                'test_acc': accs[-1],
            }

            if logger is not None:
                logger(eval_info)
            if lr_decay_factor < 1:
                scheduler.step(val_losses[-1])
            if best_acc <= current_val_acc:
                best_acc = current_val_acc
                stop_patience = 0
            else:
                stop_patience +=1

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

      print(f'seed {_} --', end=" ")
      val_acc_ = tensor(val_accs)[-(folds*epochs):].view(folds,epochs)
      val_acc_, argmax_ = val_acc_.max(dim=1)
      acc_ = tensor(accs[-(folds*epochs):]).view(folds,epochs)
      acc_ = acc_[torch.arange(folds, dtype=torch.long), argmax_]
      print('Val Acc: {:.4f}, Test Accuracy: {:.3f} +- {:.3f}'.format(val_acc_.mean().item(),
              acc_.mean().item(), acc_.std().item()))

    val_acc = tensor(val_accs)
    val_acc = val_acc.view(repeat * folds, epochs)
    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(repeat * folds, epochs)[:, :], acc.view(repeat * folds, epochs)[:,:]
    val_acc, argmax = val_acc.max(dim=1)
    acc = acc[torch.arange(repeat * folds, dtype=torch.long), argmax]
    print(argmax)
    print(acc)
    val_acc_mean = val_acc.mean().item()
    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print("[Average result]")
    print('Val Acc: {:.4f}, Test Accuracy: {:.3f} +- {:.3f}, Duration: {:.3f}'.
          format(val_acc_mean, acc_mean, acc_std, duration_mean))

    return val_acc_mean, acc_mean, acc_std



def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def backward_hook(module, grad_input, grad_output):
    global saliency
    saliency = grad_output[0].data

def drop_nodes(batch, aug_ratio=0.4):
    batch = batch.clone()
    num_nodes = batch.x.shape[0]
    num_nodes_list = scatter_add(torch.ones_like(batch.batch), batch.batch)
    cumsum_nodes = torch.cat((num_nodes_list.new_zeros(1),
        num_nodes_list[:-1])).cumsum(-1)
    k = torch.max(torch.ones_like(num_nodes_list).long().cuda(),
            (num_nodes_list * (1-aug_ratio)).long())

    node_ = torch.cat([torch.randperm(nn_.item())[:k_.item()] + cn_.item() for k_, nn_, cn_ in zip(k,
        num_nodes_list, cumsum_nodes)]).to(device)

    edge_index, _ = subgraph(node_, batch.edge_index, num_nodes=num_nodes,
                    relabel_nodes=True)
    batch.x = batch.x[node_]
    batch.edge_index = edge_index
    batch.batch = batch.batch[node_]
    return batch


def permute_edges(data, aug_ratio=0.2):
    data = data.clone()

    num_nodes_list = scatter_add(torch.ones_like(data.batch), data.batch)
    cumsum_nodes = torch.cat((num_nodes_list.new_zeros(1),
        num_nodes_list[:-1])).cumsum(-1)
    num_edges_list = (scatter_add(torch.ones_like(data.edge_index[0]),
            data.batch[data.edge_index[0]]))# / 2).long()
    cumsum_edges = torch.cat((num_edges_list.new_zeros(1),
        num_edges_list[:-1])).cumsum(-1)
    permute_num_list = (num_edges_list * aug_ratio).long()

    edge_index = data.edge_index

    idx_add = torch.cat([torch.tensor(np.random.choice(node_num.item(), (2,
        perm_num.item())))+cum_num.item() for node_num,
            cum_num, perm_num in
            zip(num_nodes_list, cumsum_nodes, permute_num_list)],
            dim=1).to(data.x.device)

    edge_index = torch.cat([edge_index[:,
        torch.randperm(edge_num.item())[:(edge_num - perm_num).item()] +
        cum_edges.item()]
        for edge_num, perm_num, cum_edges in zip(num_edges_list,
            permute_num_list, cumsum_edges)], dim=1)
    data.edge_index = torch.cat((edge_index, idx_add), dim=1)

    return data


def permute_edges_triad(data, aug_ratio=0.2):
    data = data.clone()
    num_nodes_list = scatter_add(torch.ones_like(data.batch), data.batch)
    dense_adj = to_dense_adj(data.edge_index, batch=data.batch,
            max_num_nodes=num_nodes_list.max().item())

    degree = scatter_add(data.edge_index[0], data.edge_index[1],
            dim_size=len(data.batch))
    num_nodes = len(data.batch)
    cumsum_nodes = torch.cat((num_nodes_list.new_zeros(1),
        num_nodes_list[:-1])).cumsum(-1)

    index, value = spspmm(data.edge_index, torch.ones_like(data.edge_index[0]).float(),
            data.edge_index, torch.ones_like(data.edge_index[0]).float(),
            num_nodes, num_nodes, num_nodes, coalesced=True)
    index = index[:, value != 0]
    index = index[:, index[0] != index[1]]
    b, r, c = ((1 - to_dense_adj(data.edge_index, batch=data.batch,
            max_num_nodes=num_nodes_list.max().item())) * to_dense_adj(index,
                    batch=data.batch,
                    max_num_nodes=num_nodes_list.max().item())).nonzero(as_tuple=True)
    b, r, c = b[r<c], r[r<c], c[r<c]
    assert torch.all(dense_adj[b, r, c] == 0)
    assert torch.all(dense_adj[b, c, r] == 0)


    r = r + cumsum_nodes[b]
    c = c + cumsum_nodes[b]
    assert torch.all(b == data.batch[r])
    index = torch.stack((r, c), dim=0)


    # for E_add
    common_adj = dense_adj[data.batch[r], r-cumsum_nodes[b]] * dense_adj[data.batch[r], c-cumsum_nodes[b]]
    rc_ind, common_adj = common_adj.nonzero(as_tuple=True)
    common_adj = cumsum_nodes[data.batch[r[rc_ind]]] + common_adj
    s = scatter_add(1/degree[common_adj].float(), rc_ind, dim_size=len(r))
    s_dense, s_dense_mask = to_dense_batch(s, data.batch[r])
    s_dense = s_dense / s_dense.sum(dim=1, keepdims=True)
    s_dense[~s_dense_mask] = 1e-10

    num_cand_edge = s_dense_mask.long().sum(-1)
    cumsum_cand_edge = torch.cat((num_cand_edge.new_zeros(1),
        num_cand_edge[:-1])).cumsum(-1)

    num_edges_list = scatter_add(torch.ones_like(data.edge_index[0]),
            data.batch[data.edge_index[0]]) / 2 # undirected
    cumsum_edges = torch.cat((num_edges_list.new_zeros(1),
        num_edges_list[:-1])).cumsum(-1)
    permute_num_list = (num_edges_list * aug_ratio).long()
    permute_num_list = torch.min(permute_num_list[:len(num_cand_edge)], num_cand_edge)

    while True:
        sampled = torch.multinomial(s_dense, num_samples=permute_num_list.max().item())
        sampled[permute_num_list.view(-1, 1) <=
                torch.arange(permute_num_list.max().item()).to(device)] = -1
        if torch.all(s_dense_mask[(sampled != -1).nonzero(as_tuple=True)] == 1):
            break
    add_batch, ind = (sampled != -1).nonzero(as_tuple=True)
    E_add = index[:, cumsum_cand_edge[add_batch] + ind]

    assert torch.all(data.batch[E_add[0]] == add_batch)


    k = torch.multinomial(dense_adj[add_batch, E_add[0] - cumsum_nodes[add_batch]] *
      dense_adj[add_batch, E_add[1] - cumsum_nodes[add_batch]],
      num_samples=1).squeeze(1)

    i = E_add[torch.multinomial(torch.ones_like(E_add).T.float(), num_samples=1).squeeze(1),
            torch.arange(E_add.shape[1]).to(E_add.device)] - cumsum_nodes[add_batch]

    dense_adj[add_batch, i, k] = 0
    dense_adj[add_batch, k, i] = 0


    batch_remain, r_remain, c_remain = dense_adj.nonzero(as_tuple=True)


    E_remain = torch.stack((r_remain + cumsum_nodes[batch_remain], c_remain +
        cumsum_nodes[batch_remain]), dim=0)
    assert torch.all(data.batch[E_remain[0]] == batch_remain)
    assert torch.all(data.batch[E_remain[1]] == batch_remain)
    E_add = torch.cat((E_add, torch.stack([E_add[1], E_add[0]], dim=0)), dim=1)

    edge_index = torch.cat((E_remain, E_add), dim=1)
    data.edge_index = edge_index

    return data


def mask_nodes(data, aug_ratio):
    data = data.clone()
    num_nodes_list = scatter_add(torch.ones_like(data.batch), data.batch)
    mask_num_list = (num_nodes_list * aug_ratio).long()
    cumsum_nodes = torch.cat((num_nodes_list.new_zeros(1),
        num_nodes_list[:-1])).cumsum(-1)
    node_ = torch.cat([torch.randperm(nn_.item())[:k_.item()] + cn_.item() for
        k_, nn_, cn_ in zip(mask_num_list,
        num_nodes_list, cumsum_nodes)]).to(device)
    token = data.x.mean(dim=0)
    data.x[node_] = token

    return data


def subG(data, aug_ratio):
    data = data.clone()
    batch_size = data.batch.max().item() + 1
    num_nodes = data.x.shape[0]
    num_nodes_list = scatter_add(torch.ones_like(data.batch), data.batch)
    cumsum_nodes = torch.cat((num_nodes_list.new_zeros(1),
        num_nodes_list[:-1])).cumsum(-1)
    k = torch.max(torch.ones_like(num_nodes_list).long().to(device),
            (num_nodes_list * (1-aug_ratio)).long())
    edge_index = data.edge_index
    edge_index, _ = sort_edge_index(edge_index)
    neigh, neigh_mask = to_dense_batch(edge_index[1], edge_index[0],
             fill_value=-1)
    if neigh_mask.shape[0] < num_nodes:
        neigh = torch.cat((neigh, -neigh.new_ones(num_nodes - neigh.shape[0],
            neigh.shape[1])), dim=0)
        neigh_mask = torch.cat((neigh_mask, neigh_mask.new_zeros(num_nodes -
            neigh_mask.shape[0],
            neigh_mask.shape[1])), dim=0)
    node_idx_dense, mask = \
        to_dense_batch(torch.arange(num_nodes).to(data.x.device), data.batch,
                 fill_value=-1)
    idx_sub = torch.gather(node_idx_dense, 1, torch.multinomial(mask.float(), 1)).view(-1)
    idx_sub, _ = idx_sub.sort()
    done_batch = scatter_add(torch.ones_like(idx_sub), data.batch[idx_sub],
        dim_size=batch_size) == k

    node_mask = torch.zeros(num_nodes).bool().to(data.x.device)
    node_mask[idx_sub] = True
    for _ in range(k.max() + 1):
        if torch.all(done_batch):
            break
        neighs = neigh[idx_sub][neigh_mask[idx_sub]].unique()
        neighs = neighs[~node_mask[neighs]]
        if len(neighs) == 0:
            break
        neighs, _ = torch.sort(neighs)
        neighs_batch, neighs_batch_mask = to_dense_batch(neighs, data.batch[neighs])
        no_neigh_batch = (neighs_batch_mask.long().sum(-1) == 0).nonzero(as_tuple=True)[0]
        done_batch[no_neigh_batch] = True
        neighs_batch = neighs_batch[~done_batch[:neighs_batch.shape[0]]]
        neighs_batch_mask = \
            neighs_batch_mask[~done_batch[:neighs_batch_mask.shape[0]]]
        new_neighs = torch.gather(neighs_batch, 1,
                torch.multinomial(neighs_batch_mask.float(), 1)).view(-1)
        idx_sub = torch.cat((idx_sub, new_neighs), dim=0)
        assert len(idx_sub) == len(idx_sub.unique())
        idx_sub, _ = idx_sub.sort()
        node_mask[idx_sub] = True
        done_batch = (scatter_add(torch.ones_like(idx_sub), data.batch[idx_sub],
            dim_size=batch_size) == k) | done_batch

    edge_index, _ = subgraph(idx_sub, data.edge_index, num_nodes=num_nodes,
                    relabel_nodes=True)
    data.x = data.x[idx_sub]
    data.edge_index = edge_index
    data.batch = data.batch[idx_sub]
    return data



from torch.distributions.uniform import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SigmoidTransform

base_distribution = Uniform(0, 1)
transforms = [SigmoidTransform().inv]#, AffineTransform(loc=0, scale=1)]
logistic = TransformedDistribution(base_distribution, transforms)

def bernoulli_gumbel(prob, thrs=0.5):
    Y = prob.clamp(min=1e-8).log() + logistic.sample(prob.shape).to(device)
    Y_sig = prob # torch.sigmoid(Y)
    # ST Gumbel
    return (Y_sig > thrs).float() - Y_sig.detach() + Y_sig


def label_smoothing_loss(out, y, ratio=0.1):
    K = out.shape[-1]
    loss = (1 - ratio) * F.nll_loss(out, y)
    if ratio > 0:
        for i in range(K):
            loss += ratio/K * F.nll_loss(out, torch.ones_like(y)*i)
    return loss


def evolve_filter(model, new_data, avg_class_predction, theta):
    model.eval()
    logits = None
    new_logits = model(new_data)[0]
    prob_new_logits = F.softmax(new_logits, dim=-1)
    new_sim_matrix = torch.matmul(prob_new_logits, avg_class_predction.permute(1,0))
    new_sim_vector = torch.gather(new_sim_matrix, 1, new_data.y.unsqueeze(1))
    non_filter_mask = (new_sim_vector > theta).squeeze()

    if non_filter_mask.sum() == 0:
        return None
    ### Filter augmented data ###
    non_filtered_idx = non_filter_mask[new_data.batch].nonzero(as_tuple=False).squeeze()
    new_edge_index, _ = subgraph(non_filtered_idx, new_data.edge_index,
            relabel_nodes=True, num_nodes=len(new_data.batch))
    new_data.x = new_data.x[non_filtered_idx]
    new_data.edge_index = new_edge_index
    new_data.y = new_data.y[non_filter_mask]
    if hasattr(new_data, 'edge_weight'):
        new_data.edge_weight = torch.ones_like(new_data.edge_index)

    n_batch = non_filter_mask.sum()
    new_node_list = scatter_add(torch.ones_like(new_data.batch[non_filtered_idx]),new_data.batch[non_filtered_idx])
    new_node_list = new_node_list[new_node_list!=0]
    assert len(new_node_list) == n_batch
    new_data.batch = torch.arange(n_batch).to(device).repeat_interleave(new_node_list)
    model.train()
    return new_data

def train(model, ep_net, optimizer, loader,
        method, accum_steps=1, k=1,
        ratio=0.5, manifold=False, degree_feature=False, edge_predict=True,
        edge_thrs=0.5, only_aug=False, smoothing=0.0,
        avg_class_prediction=None,theta=None):
    global saliency
    criterion_batch = nn.NLLLoss(reduction='none')
    model.train()

    total_loss = 0
    accum = 0
    n_edges = 0
    n_edges_correct = 0
    for data in loader:
        if accum == 0:
            optimizer.zero_grad()
        data = data.to(device)
        out, out_lastconv = model(data)
        if not only_aug:
            loss = label_smoothing_loss(out, data.y.view(-1), smoothing)
            (loss/accum_steps).backward() #if not only_aug else torch.zeros(1).to(device)
            total_loss += loss.item() * num_graphs(data)

        if avg_class_prediction is not None:
          for _ in range(k):
            data_aug = drop_nodes(data, ratio) if method == 'dropN' \
                    else (permute_edges(data, ratio) if method == 'permE' else
                        (subG(data, ratio) if method == 'subG' else
                            mask_nodes(data, ratio)))
            if degree_feature:
                new_degree = scatter_add(torch.ones_like(data_aug.edge_index[0]),
                        data_aug.edge_index[0],
                        dim_size=len(data_aug.x))
                data_aug.x = torch.zeros_like(data_aug.x)
                data_aug.x.scatter_(1, torch.min(new_degree[:, None],
                    torch.ones_like(new_degree[:, None]) *
                    (data_aug.x.shape[1]-1)),
                        data_aug.x.new_ones(len(data_aug.x), 1))
            if data_aug is None:
                continue
            batch_size = data.batch.max().item() + 1

            out_aug, out_lastconv_aug = model(data_aug)

            loss_aug = label_smoothing_loss(out_aug, data_aug.y.view(-1),
                    smoothing) / k
            (len(data_aug.y) / batch_size * loss_aug/accum_steps).backward()

            total_loss += loss_aug.item() / k * num_graphs(data)
        accum += 1
        if (accum + 1) == accum_steps:
            accum = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader):
    model.eval()

    correct = 0
    logits = None
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            logit = model(data)[0]
            pred = logit.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()

        if logits == None:
            logits = logit
            labels = data.y.view(-1)
            correct_mask = pred.eq(data.y.view(-1))
        else:
            logits = torch.cat((logits, logit),dim = 0)
            labels = torch.cat((labels, data.y.view(-1)), dim = 0)
            correct_mask = torch.cat((correct_mask,pred.eq(data.y.view(-1))),dim = 0)

    ### Calculate threshold theta ###
    n_class = logits.shape[1]
    class_num_list = scatter_add(torch.ones_like(labels), labels, dim_size = n_class)
    prob_logits = F.softmax(logits, dim=-1)
    avg_class_predction = torch.div(scatter_add(prob_logits, labels, dim=0), class_num_list)
    assert avg_class_predction.shape == (n_class, n_class)

    sim_matrix = torch.matmul(prob_logits, avg_class_predction.permute(1,0)) # [n_graph, n_class]
    sim_vector = torch.gather(sim_matrix, 1, labels.unsqueeze(1)) # [n_graph,]

    theta_space = torch.linspace(0, 1, 50).to(device)
    sigma_correct = (theta_space.unsqueeze(1) > sim_vector[correct_mask].squeeze()).sum(dim=1)
    sigma_wrong = (sim_vector[~correct_mask].squeeze() > theta_space.unsqueeze(1)).sum(dim=1)
    sigma = sigma_correct + sigma_wrong
    theta = theta_space[torch.argmin(sigma)]

    return correct / len(loader.dataset), avg_class_predction, theta


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out, _ = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)

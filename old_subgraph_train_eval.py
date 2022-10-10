import time


from sklearn.model_selection import StratifiedKFold, train_test_split
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import to_dense_adj, to_dense_batch, subgraph, \
        sort_edge_index
from utils import k_hop_subgraph
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

saliency = None

import copy

def save_checkpoint(state, expname, fold, filename='model_best.pth.tar'):
    directory = "runs/%s_%s/" % (expname, str(fold))
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)



def cross_validation_with_val_set(dataset, model, ep_net, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, method, logger=None, k=1,
                                  ratio=0.5, edge_predict=True,
                                  edge_thrs=0.5, expname=None,
                                  R=0.1,
                                  train_reduce=1):
    model.convs[-1].register_backward_hook(backward_hook)
    val_accs, val_losses, accs, durations = [], [], [], []
    eces = []
    repeat = 3
    for _ in range(repeat):
        for fold, (train_idx, test_idx,
                   val_idx) in enumerate(zip(*k_fold(dataset, folds,
                       train_reduce))):
            torch.cuda.empty_cache()
            dataset_ = copy.deepcopy(dataset)
            if dataset.num_node_attributes > 0:
                n_attr = dataset.num_node_attributes
                mean = dataset.data.x[torch.cat((train_idx, val_idx))][:, :n_attr].mean(dim=0,
                        keepdims=True)
                std = dataset.data.x[torch.cat((train_idx, val_idx))][:, :n_attr].std(dim=0,
                        keepdims=True)
                dataset_.data.x[:, :n_attr] -= mean
                dataset_.data.x[:, :n_attr] /= std

            train_dataset = dataset_[train_idx]
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
            iters = len(train_loader)
            if lr_decay_factor < 1:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                 factor=lr_decay_factor,
                                                                 patience=int(1000/iters),
                                                                 verbose=False)

            t_start = time.perf_counter()
            stop_patience=0
            best_acc = 0
            best_ece = 0
            for epoch in tqdm(range(1, epochs + 1), ncols=50):
                if stop_patience == int(1500/iters):
                    train_loss = -1
                    val_losses.append(100)
                    val_accs.append(-1)
                    accs.append(-1)
                    continue
                else:
                    train_loss = train(model, ep_net, optimizer, train_loader, method,
                            accum_steps=max(int(64/batch_size), 1), ratio=ratio,
                            degree_feature=dataset.degree_feature,
                            edge_predict=edge_predict, edge_thrs=edge_thrs,
                            R=R)
                    val_losses.append(eval_loss(model, val_loader))
                    current_val_acc = eval_acc(model, val_loader)[0]
                    val_accs.append(current_val_acc)
                    current_test_acc, current_ece = eval_acc(model, test_loader)
                    accs.append(current_test_acc)

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


                # Early stopping
                if best_acc <= current_val_acc:
                    best_acc = current_val_acc
                    best_ece = current_ece
                    stop_patience = 0
                else:
                    stop_patience +=1


            eces.append(float(best_ece))
            t_end = time.perf_counter()
            durations.append(t_end - t_start)

        print(f'seed {_} --', end=" ")
        val_acc_ = tensor(val_accs)[-(folds*epochs):].view(folds,epochs)
        val_acc_, argmax_ = val_acc_.max(dim=1)
        acc_ = tensor(accs[-(folds*epochs):]).view(folds,epochs)
        acc_ = acc_[torch.arange(folds, dtype=torch.long), argmax_]
        print('Val Acc: {:.4f}, Test Accuracy: {:.3f} +- {:.3f}'.format(val_acc_.mean().item(),
                acc_.mean().item(), acc_.std().item()))

        print(f'ECE: {eces}')
        print(f'MeanECE: {sum(eces) / len(eces)}')
        print(f'ECE_STD: {statistics.pstdev(eces)}')
    val_acc = tensor(val_accs)
    val_acc = val_acc.view(repeat*folds,epochs)
    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(repeat*folds, epochs)[:,:], acc.view(repeat*folds, epochs)[:,:]
    val_acc, argmax = val_acc.max(dim=1)
    acc = acc[torch.arange(repeat*folds, dtype= torch.long),argmax]
    val_acc_mean = val_acc.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print("[Average result]")
    print('Val Acc: {:.4f}, Test Accuracy: {:.3f} +- {:.3f}, Duration: {:.3f}'.
          format(val_acc_mean, acc_mean, acc_std, duration_mean))

    return val_acc_mean, acc_mean, acc_std


def k_fold(dataset, folds, train_reduce=1):
    # train_reduce : k 이면 1/k 하는 것
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
        if train_reduce > 1:
            sub = train_test_split(train_indices[-1],
                    stratify=dataset.data.y[train_indices[-1]],
                    random_state=12345, test_size=1/train_reduce)[1]
            train_indices[-1] = sub
    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def backward_hook(module, grad_input, grad_output):
    global saliency
    saliency = grad_output[0].data


from torch.distributions.uniform import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SigmoidTransform

base_distribution = Uniform(0, 1)
transforms = [SigmoidTransform().inv]#, AffineTransform(loc=0, scale=1)]
logistic = TransformedDistribution(base_distribution, transforms)

def bernoulli_gumbel(prob, thrs=0.5):
    Y = prob.clamp(min=1e-8).log() + logistic.sample(prob.shape).to(device)
    Y_sig = torch.sigmoid(Y)
    # ST Gumbel
    return (Y_sig > thrs).float() - Y_sig.detach() + Y_sig


def train(model, ep_net, optimizer, loader, method, accum_steps=1,
        ratio=0.5, degree_feature=False, edge_predict=True,
        edge_thrs=0.5, R=0.1):
    salient_src = True
    prenormalize = True
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
        batch_size = data.batch.max().item() + 1

        k = random.choice([1,2,3])
        out, out_lastconv = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        (loss/accum_steps).backward()

        if method == 'vanilla':
            total_loss += loss.item() * num_graphs(data)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()

        elif method == 'graphtransplant':
            num_nodes = data.x.shape[0]
            out_lastconv = out_lastconv.detach()
            ### Training the edge predictor ###
            # edge 있음
            if edge_predict:
                num_subedges = min(num_nodes, data.edge_index.shape[1])
                subset = torch.randperm(data.edge_index.shape[1])[:num_subedges]
                row_sub, col_sub = data.edge_index[:, subset]
                pair_1 = torch.cat((out_lastconv[row_sub],
                    out_lastconv[col_sub]), dim=1)
                label_1 = torch.ones(num_subedges).long().to(device)

                # edge 없음
                ind, mask = to_dense_batch(torch.arange(data.x.shape[0]).to(device), data.batch,
                        fill_value=-1) # batch x max_num_nodes
                ind = ind[data.batch] # num_nodes x max_num_nodes
                cand = ind > -1
                num_nodes_list = scatter_add(torch.ones_like(data.batch), data.batch)
                cumsum_nodes = torch.cat((num_nodes_list.new_zeros(1),
                    num_nodes_list[:-1])).cumsum(-1)
                cand[data.edge_index[0], data.edge_index[1] -
                        cumsum_nodes[data.batch[data.edge_index[1]]]] = False
                pair = torch.multinomial(cand[row_sub].float(), 1)
                ind = torch.gather(ind[row_sub], 1,
                        pair).squeeze(-1)
                label_0 = torch.zeros(num_subedges).long().to(device)
                pair_0 = torch.cat((out_lastconv[row_sub],
                    out_lastconv[ind]), dim=1)

                edge_logits = ep_net(torch.cat((pair_0, pair_1), dim=0)).squeeze(-1)
                label = torch.cat((label_0, label_1), dim=0)
                n_edges_correct += ((torch.sigmoid(edge_logits) > 0.5).long() ==
                        label).long().sum().item()
                n_edges += len(edge_logits)
                ep_loss = F.binary_cross_entropy_with_logits(edge_logits, label.float())
                (ep_loss/accum_steps).backward()
            ### CutMix ###
            # Saliency information (L2 norm)
            # Store the index of the starting node of each graph
            num_nodes_list = scatter_add(torch.ones_like(data.batch), data.batch)
            cumsum_nodes = idx = torch.cat((num_nodes_list.new_zeros(1),
                num_nodes_list[:-1])).cumsum(-1)

            # Determine the src-dst pairs
            # TODO: Stochastic sampling
            rand_index = torch.randperm(data.num_graphs).to(device) # tgt -> src[rand_index]
            rand_index_inv = rand_index.argsort()
            ## CHANGE ##
            k_src = torch.max(torch.ones_like(num_nodes_list).long().to(device),
                    (num_nodes_list[rand_index] * R).long())
            k_dst = torch.max(torch.ones_like(num_nodes_list).long().to(device),
                    (num_nodes_list * R).long())

            new_data = data.clone()
            label_weight = torch.zeros_like(data.y).float()


            if salient_src:
                saliency = torch.sqrt(torch.sum(saliency ** 2, dim = -1)) + 1e-10
                saliency_batch = scatter_add(saliency, data.batch,
                        dim_size=batch_size)
                if prenormalize:
                    saliency = saliency / saliency_batch[data.batch]
                size_based_saliency = torch.ones_like(data.batch).float() / num_nodes_list[data.batch].float()
                dense_saliency, dense_mask = to_dense_batch(saliency + 1e-10, data.batch)
                topk_idx = dense_saliency.argsort(dim=-1, descending=True) + idx.view(-1, 1)
                topk_idx[~dense_mask] = -1
                batch_ = rand_index.repeat_interleave(k_src)

                k_src_prob = torch.ones(len(k_src), 2*k_src.max().item()).to(device)
                k_src_prob[2*k_src.view(-1, 1) <=
                        torch.arange(2*k_src.max().item()).to(k_src.device)] = 1e-10
                while True:
                    sampled = torch.multinomial(k_src_prob, k_src.max().item())
                    sampled[k_src.view(-1, 1) <=
                            torch.arange(k_src.max().item()).to(k_src.device)] = -1
                    if torch.all(k_src_prob[(sampled != -1).nonzero(as_tuple=True)] == 1):
                        break
                node_ = sampled[sampled > -1]
                topk_idx = topk_idx[batch_, node_]
                assert torch.all(topk_idx != -1)
            else:
                k_src_prob = torch.ones(len(k_src), num_nodes_list.max().item()).to(device)
                k_src_prob[num_nodes_list[rand_index].view(-1, 1) <=
                        torch.arange(num_nodes_list.max().item()).to(num_nodes_list.device)] = 1e-10
                while True:
                    sampled = torch.multinomial(k_src_prob, k_src.max().item())
                    sampled[k_src.view(-1, 1) <=
                            torch.arange(k_src.max().item()).to(k_src.device)] = -1
                    if torch.all(k_src_prob[(sampled != -1).nonzero(as_tuple=True)] == 1):
                        break
                batchind, _ = (sampled > -1).nonzero(as_tuple=True)
                topk_idx = sampled[sampled > -1] + cumsum_nodes[rand_index[batchind]]

            # TODO randomly selected subgraph
            beta = torch.distributions.beta.Beta(2,2)
            ratio = beta.sample()
            sub_node, edge_index_src, inv_src, _ = k_hop_subgraph(node_idx=topk_idx,
                    num_hops=k, edge_index= data.edge_index, num_nodes =num_nodes,
                   relabel_nodes=True, ratio=ratio,
                   batch=data.batch)

            degree = scatter_add(torch.ones_like(data.edge_index[0]),
                    data.edge_index[0], dim_size=len(data.x))
            src_sub_degree = scatter_add(torch.ones_like(edge_index_src[0]),
                    edge_index_src[0], dim_size=len(sub_node))
            if salient_src:
                src_sub_saliency = scatter_add(saliency[sub_node],
                        data.batch[sub_node], dim_size=batch_size)
            src_changed_node_bool = (degree[sub_node] !=
                    src_sub_degree)
            src_changed_node = src_changed_node_bool.nonzero(as_tuple=True)[0]
            out_lastconv_src = out_lastconv[sub_node]
            x_src = data.x[sub_node]

            batch_src = data.batch[sub_node]
            new_batch_src = rand_index_inv[batch_src]

            dense_src_changed_node, _ = to_dense_batch(src_changed_node, batch_src[src_changed_node],
                    fill_value=-1) if len(src_changed_node) else (src_changed_node.new_ones(0, 0), None)
            if dense_src_changed_node.shape[0] != batch_size:
                dense_src_changed_node = torch.cat((dense_src_changed_node,
                    - dense_src_changed_node.new_ones(batch_size -
                        dense_src_changed_node.shape[0],
                        dense_src_changed_node.shape[1])), dim=0)

            k_dst_prob = torch.ones(len(k_dst), num_nodes_list.max().item()).to(device)
            k_dst_prob[num_nodes_list.view(-1, 1) <=
                    torch.arange(num_nodes_list.max().item()).to(num_nodes_list.device)] = 1e-10
            while True:
                sampled = torch.multinomial(k_dst_prob, k_dst.max().item())
                sampled[k_dst.view(-1, 1) <=
                        torch.arange(k_dst.max().item()).to(k_dst.device)] = -1
                if torch.all(k_dst_prob[(sampled != -1).nonzero(as_tuple=True)] == 1):
                    break

            batchind, _ = (sampled > -1).nonzero(as_tuple=True)
            dst_idx = sampled[sampled > -1] + cumsum_nodes[batchind]
            #Remove dst k_hop_subgraph
            remove_idx, _, inv_tgt , _ = k_hop_subgraph(node_idx=dst_idx, num_hops=k,
                                   edge_index= data.edge_index, num_nodes=num_nodes,
                                   relabel_nodes=True, ratio=ratio,
                                   batch=data.batch)
            remain = torch.ones(num_nodes).to(device).bool()
            remain[remove_idx] = False
            x_dst = data.x[remain]
            edge_index_src = edge_index_src + len(x_dst)

            out_lastconv_dst = out_lastconv[remain]
            out_lastconv = torch.cat((out_lastconv_dst, out_lastconv_src),
                    dim=0)
            batch_dst = data.batch[remain]
            new_batch = torch.cat((batch_dst, new_batch_src), dim=0)
            # dst subgraph
            edge_index_dst, _ = subgraph(remain, data.edge_index, num_nodes=num_nodes,
                    relabel_nodes=True)
            new_dst_ind = torch.arange(len(x_dst)).to(device)
            dst_sub_degree = scatter_add(torch.ones_like(edge_index_dst[0]),
                    edge_index_dst[0], dim_size=len(x_dst))
            dst_changed_node = (degree[remain] !=
                    dst_sub_degree).nonzero(as_tuple=True)[0]
            sum_change_src = scatter_add(torch.ones_like(src_changed_node),
                    batch_src[src_changed_node], dim_size=batch_size)
            sum_change_dst = scatter_add(torch.ones_like(dst_changed_node),
                    batch_dst[dst_changed_node], dim_size=batch_size)
            edge_dst = dst_changed_node.repeat_interleave(sum_change_src[rand_index].repeat_interleave(sum_change_dst))
            edge_src = dense_src_changed_node[rand_index].repeat_interleave(sum_change_dst,
                dim=0).view(-1)
            edge_src = edge_src[edge_src!=-1] + len(x_dst)
            assert len(edge_dst) == len(edge_src)
            if salient_src:
                dst_sub_saliency = scatter_add(saliency[remain],
                        data.batch[remain], dim_size=batch_size)
            sum_sub_src = scatter_add(torch.ones_like(batch_src), batch_src,
                    dim_size=batch_size)
            sum_sub_dst = scatter_add(torch.ones_like(batch_dst), batch_dst,
                    dim_size=batch_size)

            if edge_predict:
                edge_pred_logit_s2d = ep_net(torch.cat((out_lastconv[edge_src],
                    out_lastconv[edge_dst]), dim=1)).squeeze(-1)
                edge_pred_logit_d2s = ep_net(torch.cat((out_lastconv[edge_dst],
                    out_lastconv[edge_src]), dim=1)).squeeze(-1)

                edge_prob = (torch.sigmoid(edge_pred_logit_d2s) +
                    torch.sigmoid(edge_pred_logit_s2d)) / 2
                edge_weight_new = bernoulli_gumbel(edge_prob, thrs=edge_thrs)

            # TODO new_data
            edge_weight = torch.ones(edge_index_dst.shape[1] +
                    edge_index_src.shape[1]).to(device)
            edge_index = torch.cat((edge_index_dst, edge_index_src,
                ),
                dim=1)
            if edge_predict:
                edge_weight = torch.cat((edge_weight,
                    edge_weight_new,
                    edge_weight_new
                    ),
                    dim=0) # undirected graph
                edge_index = torch.cat((edge_index,
                    torch.stack((edge_dst, edge_src), dim=0),
                    torch.stack((edge_src, edge_dst), dim=0)
                    ),
                    dim=1)
            new_data.x = torch.cat((x_dst, x_src), dim=0)
            if degree_feature:
                new_degree = scatter_add(edge_weight.long(),edge_index[0],
                        dim_size=len(new_data.x))
                new_data.x = torch.zeros_like(new_data.x)
                new_data.x.scatter_(1, torch.min(new_degree[:, None],
                    torch.ones_like(new_degree[:, None]) *
                    (new_data.x.shape[1]-1)),
                        new_data.x.new_ones(len(new_data.x), 1))

            new_data.edge_index = edge_index
            new_data.edge_weight = edge_weight
            new_data.batch = torch.cat((batch_dst, new_batch_src), dim=0)

            # Saliency-based label
            if salient_src:
                if not prenormalize:
                    imp_src = (src_sub_saliency / saliency_batch)[rand_index]
                    imp_dst = (dst_sub_saliency / saliency_batch)
                else:
                    imp_src = (src_sub_saliency)[rand_index]
                    imp_dst = (dst_sub_saliency)
            else: # size-based label
                imp_src = (sum_sub_src.float() /
                        num_nodes_list.float())[rand_index]
                imp_dst = sum_sub_dst.float() / num_nodes_list.float()

            label_weight = imp_src / (imp_src + imp_dst)

            #Augmented data training
            new_out, out_lastconv = model(data=new_data)
            augmented_loss = (1-label_weight) * criterion_batch(new_out, data.y.view(-1)) + label_weight * criterion_batch(new_out, data.y[rand_index].view(-1))
            augmented_loss = torch.mean(augmented_loss)
            (augmented_loss/accum_steps).backward()
            total_loss += (loss.item() + augmented_loss.item()) * num_graphs(data)
            accum += 1
            if accum == accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
                accum = 0
    if accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
    return total_loss / len(loader.dataset)

import scipy.sparse as sp


def eval_acc(model, loader):
    ece_criterion = ECE()
    model.eval()
    logits = None
    correct = 0
    for data in loader:
        data = data.to(device)
        '''
        dense_adj = to_dense_adj(data.edge_index).squeeze().cpu().numpy()

        dist = sp.csgraph.shortest_path(dense_adj)
        num_nodes_list = scatter_add(torch.ones_like(data.batch), data.batch)
        cumsum_nodes = torch.cat((num_nodes_list.new_zeros(1),
                                  num_nodes_list[:-1])).cumsum(-1)

        num_edges_list = scatter_add(torch.ones_like(data.edge_index[0]),
                                     data.batch[data.edge_index[0]])

        cumsum_edges = torch.cat((num_edges_list.new_zeros(1),
                                  num_edges_list[:-1])).cumsum(-1)

        permute_num = np.random.choice(range(10,16), len(num_edges_list), replace=True)
        permute_num_list = torch.min(num_edges_list ,torch.tensor(permute_num).to(device))


        edge_index = data.edge_index
        # Add edges.
        idx_add = None
        for node_num, cum_num, perm_num in zip(num_nodes_list, cumsum_nodes, permute_num_list):
            perm_num /= 2
            temp_dist = np.triu(dist[cum_num:cum_num+node_num, cum_num:cum_num +node_num])
            x_idx , y_idx = np.where( (temp_dist< 5) & (temp_dist > 0) )
            x_idx , y_idx = torch.tensor(x_idx).unsqueeze(0), torch.tensor(y_idx).unsqueeze(0)
            add_idx = torch.cat((x_idx, y_idx), dim=0).to(data.x.device)
            rand_idx = torch.randperm(add_idx.shape[1])[:perm_num]
            add_idx = add_idx[:,rand_idx] + cum_num.item()
            if idx_add == None:
                idx_add = add_idx
            else:
                idx_add = torch.cat((idx_add, add_idx),dim=1)
        edge_index = torch.cat([edge_index[:,
            torch.randperm(edge_num.item())[:(edge_num - perm_num//2).item()] +
            cum_edges.item()]
            for edge_num, perm_num, cum_edges in zip(num_edges_list,
                permute_num_list, cumsum_edges)], dim=1)
        data.edge_index = torch.cat((edge_index, idx_add), dim = 1)
        '''
        with torch.no_grad():
            pred, _ = model(data)
            pred = pred.max(1)[1]
            #pred = model(data)[0].max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        '''
        if logits == None:
            logits = logit
            labels = data.y.view(-1)
        else:
            logits = torch.cat((logits, logit),dim = 0)
            labels = torch.cat((labels, data.y.view(-1)), dim = 0)
        '''
    #ece_value = ece_criterion(logits, labels)
    return correct / len(loader.dataset) , 0#ece_value


def eval_loss(model, loader):
    model.eval()
    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out, _  = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)

class ECE(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        # softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


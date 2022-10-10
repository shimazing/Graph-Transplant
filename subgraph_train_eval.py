import time


from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch, to_dense_adj, subgraph
from utils import k_hop_subgraph
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

saliency = None

import copy
def cross_validation_with_val_set(dataset, model, ep_net, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, method, logger=None, k=1,
                                  ratio=0.5, manifold=False, edge_predict=True,
                                  edge_thrs=0.5, R=0.1, smoothing=0.0,
                                  layer_sal=False, matching=False,
                                  use_sal=False,
                                  train_reduce=1):
    model.convs[-1].register_backward_hook(backward_hook)
    val_accs, val_losses, accs, durations = [], [], [], []
    repeat = 3
    for _ in range(repeat):
        if False:#isinstance(dataset, list):
            folds = 1
            train_dataset, val_dataset, test_dataset = dataset
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
                                                                 #params['lr_reduce_factor'],
                                                                 patience=20, #int(1000/iters),
                                                                 #params['lr_schedule_patience'],
                                                                 verbose=False)
            #if torch.cuda.is_available():
            #    torch.cuda.synchronize()

            t_start = time.perf_counter()
            stop_patience=0
            best_acc = 0
            for epoch in tqdm(range(1, epochs + 1), ncols=50):
                if stop_patience == 50: #int(15000/iters):
                    train_loss = -1
                    val_losses.append(100)
                    val_accs.append(-1)
                    accs.append(-1)
                    continue
                else:
                    train_loss = train(model, ep_net, optimizer, train_loader, method,
                            accum_steps=max(int(64/batch_size), 1), k=k, ratio=ratio,
                            manifold=manifold, degree_feature=train_dataset.degree_feature,
                            edge_predict=edge_predict, edge_thrs=edge_thrs,
                            smoothing=smoothing)
                    val_losses.append(eval_loss(model, val_loader))
                    current_val_acc = eval_acc(model, val_loader)
                    val_accs.append(current_val_acc)
                    accs.append(eval_acc(model, test_loader))
                    print("epoch", epoch, val_accs[-1], accs[-1])
                    eval_info = {
                        'fold':  0,
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

                #if epoch % lr_decay_step_size == 0:
                #    for param_group in optimizer.param_groups:
                #        param_group['lr'] = lr_decay_factor * param_group['lr']

                # Early stopping
                if best_acc <= current_val_acc:
                    best_acc = current_val_acc
                    stop_patience = 0
                else:
                    stop_patience +=1



            t_end = time.perf_counter()
            durations.append(t_end - t_start)

        else:
          for fold, (train_idx, test_idx,
                   val_idx) in enumerate(zip(*k_fold(dataset, folds))):
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
                                                                 #params['lr_reduce_factor'],
                                                                 patience=int(1000/iters),
                                                                 #params['lr_schedule_patience'],
                                                                 verbose=False)
            #if torch.cuda.is_available():
            #    torch.cuda.synchronize()

            t_start = time.perf_counter()
            stop_patience=0
            best_acc = 0
            for epoch in tqdm(range(1, epochs + 1), ncols=50):
                if stop_patience == int(1500/iters):
                    train_loss = -1
                    val_losses.append(100)
                    val_accs.append(-1)
                    accs.append(-1)
                    continue
                else:
                    train_loss = train(model, ep_net, optimizer, train_loader, method,
                            accum_steps=max(int(64/batch_size), 1), k=k, ratio=ratio,
                            manifold=manifold, degree_feature=dataset.degree_feature,
                            edge_predict=edge_predict, edge_thrs=edge_thrs, R=R,
                            smoothing=smoothing)
                    val_losses.append(eval_loss(model, val_loader))
                    current_val_acc = eval_acc(model, val_loader)
                    val_accs.append(current_val_acc)
                    accs.append(eval_acc(model, test_loader))

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

                #if epoch % lr_decay_step_size == 0:
                #    for param_group in optimizer.param_groups:
                #        param_group['lr'] = lr_decay_factor * param_group['lr']

                # Early stopping
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
    val_acc = val_acc.view(repeat*folds,epochs)
    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(repeat*folds, epochs)[:,:], acc.view(repeat*folds, epochs)[:,:]
    #loss, argmin = loss.min(dim=1)
    val_acc, argmax = val_acc.max(dim=1)
    acc = acc[torch.arange(repeat*folds, dtype= torch.long),argmax]
    #print(argmax)
    #print(acc)
    val_acc_mean = val_acc.mean().item()
    #loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    #print('Val Loss: {:.4f}, Test Accuracy: {:.3f} +- {:.3f}, Duration: {:.3f}'.
    print("[Average result]")
    print('Val Acc: {:.4f}, Test Accuracy: {:.3f} +- {:.3f}, Duration: {:.3f}'.
          format(val_acc_mean, acc_mean, acc_std, duration_mean))

    #return loss_mean, acc_mean, acc_std
    return val_acc_mean, acc_mean, acc_std


def k_fold(dataset, folds):
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

def label_smoothing_loss(out, y, ratio=0.1):
    K = out.shape[-1]
    loss = (1 - ratio) * F.nll_loss(out, y)
    if ratio > 0:
        for i in range(K):
            loss += ratio/K * F.nll_loss(out, torch.ones_like(y)*i)
    return loss


def train(model, ep_net, optimizer, loader, method, accum_steps=1, k=1,
        ratio=0.5, manifold=False, degree_feature=False, edge_predict=True,
        edge_thrs=0.5, R=0.1, smoothing=0.0):
    ###GraphMIX###
    global saliency
    criterion_batch = nn.NLLLoss(reduction='none')
    model.train()

    total_loss = 0
    accum = 0
    #accum_steps = 32
    n_edges = 0
    n_edges_correct = 0
    for data in loader:
      #with torch.autograd.detect_anomaly():
        if accum == 0:
            optimizer.zero_grad()
        data = data.to(device)
        batch_size = data.batch.max().item() + 1

        #manifold mixup
        if manifold:
            manifold_layer = torch.randint(model.num_layers+1, (1,)).item()
            out, out_lastconv, manifold_feature  = model(data=data, manifold_layer=manifold_layer)
            manifold_feature = manifold_feature.detach()
        else:
            out, out_lastconv = model(data)
            manifold_layer = -1
        #loss = F.nll_loss(out, data.y.view(-1))
        loss = label_smoothing_loss(out, data.y.view(-1), smoothing)
        (loss/accum_steps).backward()#retain_graph=True)
        if method == 'vanilla':
            total_loss += loss.item() * num_graphs(data)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()

        elif 'graphmix' in method:
            num_nodes = data.x.shape[0]
            out_lastconv = out_lastconv.detach()
            ### Training the edge predictor ###
            # edge 있음
            if ('randconn' not in method) and edge_predict:
                num_subedges = min(num_nodes, data.edge_index.shape[1])
                subset = torch.randperm(data.edge_index.shape[1])[:num_subedges]
                row_sub, col_sub = data.edge_index[:, subset]
                pair_1 = torch.cat((out_lastconv[row_sub],
                    out_lastconv[col_sub]), dim=1)
                label_1 = torch.ones(num_subedges).long().to(device)
                #print("label_1", label_1)

                # edge 없음
                ind, mask = to_dense_batch(torch.arange(data.x.shape[0]).to(device), data.batch,
                        fill_value=-1) # batch x max_num_nodes
                ind = ind[data.batch] # num_nodes x max_num_nodes

                cand = ind > -1
                #cand = torch.where(ind >= 0,
                #        ind + torch.arange(num_nodes).to(device).view(-1, 1) *
                #        num_nodes, -torch.ones_like(ind))
                num_nodes_list = scatter_add(torch.ones_like(data.batch), data.batch)
                cumsum_nodes = torch.cat((num_nodes_list.new_zeros(1),
                    num_nodes_list[:-1])).cumsum(-1)
                cand[data.edge_index[0], data.edge_index[1] -
                        cumsum_nodes[data.batch[data.edge_index[1]]]] = False
                pair = torch.multinomial(cand[row_sub].float(), 1)

                #cand[cand.sum(-1) == 0] = 1
                #pair =  torch.multinomial((ind[data.batch[data.edge_index[0]]] >= 0).float(), 1)
                ind = torch.gather(ind[row_sub], 1,
                        pair).squeeze(-1)
                label_0 = torch.zeros(num_subedges).long().to(device)
                #torch.tensor((np.in1d((data.edge_index[0] * num_nodes + ind).cpu(),
                #    edge_index.cpu())) | \
                #    (np.in1d((ind * num_nodes + data.edge_index[0]).cpu(),
                #        edge_index.cpu()))).long().to(device)
                #print("label_0", label_0)
                #input()
                pair_0 = torch.cat((out_lastconv[row_sub],
                    out_lastconv[ind]), dim=1)

                edge_logits = ep_net(torch.cat((pair_0, pair_1), dim=0)).squeeze(-1)
                label = torch.cat((label_0, label_1), dim=0)
                n_edges_correct += ((torch.sigmoid(edge_logits) > 0.5).long() ==
                        label).long().sum().item()
                n_edges += len(edge_logits)
                #print(n_edges_correct, n_edges)
                #print(label.float().mean())
                ep_loss = F.binary_cross_entropy_with_logits(edge_logits, label.float())
                (ep_loss/accum_steps).backward()#retain_graph=True)

                #num_subedges = min(num_nodes, data.edge_index.shape[1])
                ##subset = torch.randperm(data.edge_index.shape[1])[:num_subedges]
                #subset = np.random.choice(data.edge_index.shape[1], num_subedges)
                #row_sub, col_sub = data.edge_index[:, subset]
                #pair_1 = torch.cat((out_lastconv[row_sub],
                #    out_lastconv[col_sub]), dim=1)
                #label_1 = torch.ones(num_subedges).long().to(device)
                ##print("label_1", label_1)
                ## edge 없음
                #adj = to_dense_adj(data.edge_index,
                #        max_num_nodes=num_nodes).squeeze(0)
                #cand = ((adj != 1) & (data.batch == data.batch.view(-1, 1))).float()
                #ind = torch.multinomial(cand[row_sub], 1).view(-1)
                #label_0 = torch.zeros(num_subedges).long().to(device)
                #pair_0 = torch.cat((out_lastconv[row_sub],
                #    out_lastconv[ind]), dim=1)
                #edge_logits = ep_net(torch.cat((pair_0, pair_1), dim=0)).squeeze(-1)
                #label = torch.cat((label_0, label_1), dim=0)
                #n_edges_correct += ((torch.sigmoid(edge_logits) > 0.5).long() ==
                #        label).long().sum().item()
                #n_edges += len(edge_logits)
                ##print(n_edges_correct, n_edges)
                ##print(label.float().mean())
                #ep_loss = F.binary_cross_entropy_with_logits(edge_logits, label.float())
                #(ep_loss/accum_steps).backward()#retain_graph=True)

            ### CutMix ###
            # Saliency information (L2 norm)
            saliency = torch.sqrt(torch.sum(saliency ** 2, dim = -1)) + 1e-8
            saliency_batch = scatter_add(saliency, data.batch,
                    dim_size=batch_size)
            # Store the index of the starting node of each graph
            num_nodes_list = scatter_add(torch.ones_like(data.batch), data.batch)
            cumsum_nodes = idx = torch.cat((num_nodes_list.new_zeros(1),
                num_nodes_list[:-1])).cumsum(-1)

            # Determine the src-dst pairs
            # TODO: Stochastic sampling
            rand_index = torch.randperm(data.num_graphs).to(device) # dst index -> src_index
            rand_index_inv = rand_index.argsort() # src index -> dst index
            ## CHANGE ##
            k_src = torch.max(torch.ones_like(num_nodes_list).long().to(device),
                    (num_nodes_list[rand_index] * R).long())
            k_dst = torch.max(torch.ones_like(num_nodes_list).long().to(device),
                    (num_nodes_list * R).long())

            new_data = data.clone()
            label_weight = torch.zeros_like(data.y).float()

            dense_saliency, dense_mask = to_dense_batch(saliency + 1e-10, data.batch)

            salient_src = True
            if salient_src:
                topk_idx_dst = topk_idx = dense_saliency.argsort(dim=-1, descending=True) + idx.view(-1, 1)
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
                #batchind, _ = (sampled > 0).nonzero(as_tuple=True)
                node_ = sampled[sampled > -1] # + cumsum_nodes[batchind]
                #node_ = torch.cat([torch.randperm(2*k_)[:k_.item()] for k_ in k_src]).to(device)
                topk_idx = topk_idx[batch_, node_]
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

            beta = torch.distributions.beta.Beta(2,2)
            ratio = beta.sample()
            k = random.choice([1,2,3])
            # Construct the k-hop subgraph
            sub_node, edge_index_src, _ , _ = k_hop_subgraph(node_idx=topk_idx,
                    num_hops=k, edge_index= data.edge_index, num_nodes =num_nodes,
                   relabel_nodes=True, ratio=ratio)
            degree = scatter_add(torch.ones_like(data.edge_index[0]),
                    data.edge_index[0], dim_size=len(data.x))
            src_sub_degree = scatter_add(torch.ones_like(edge_index_src[0]),
                    edge_index_src[0], dim_size=len(sub_node))
            src_sub_saliency = scatter_add(saliency[sub_node],
                    data.batch[sub_node], dim_size=batch_size)
            src_changed_node = (degree[sub_node] !=
                    src_sub_degree).nonzero(as_tuple=True)[0]
            if 'whole' in method:
                src_changed_node = (degree[sub_node] >=
                        0).nonzero(as_tuple=True)[0]
            src_num_changed_node = scatter_add(degree[sub_node] -
                    src_sub_degree, data.batch[sub_node],
                    dim_size=batch_size)[rand_index]
            '''
            # Top-k src subgraph
            edge_index_src, _ = subgraph(topk_idx, data.edge_index, num_nodes=num_nodes,
                    relabel_nodes=True)
            '''
            out_lastconv_src = out_lastconv[sub_node] #out_lastconv[topk_idx]
            if manifold:
                x_src = manifold_feature[sub_node]
            else:
                x_src = data.x[sub_node] #data.x[topk_idx]

            batch_src = data.batch[sub_node]
            new_batch_src = rand_index_inv[batch_src]

            dense_src_changed_node, _ = to_dense_batch(src_changed_node, batch_src[src_changed_node],
                    fill_value=-1) if len(src_changed_node) else (src_changed_node.new_ones(0, 0), None)
            if dense_src_changed_node.shape[0] != batch_size:
                dense_src_changed_node = torch.cat((dense_src_changed_node,
                    - dense_src_changed_node.new_ones(batch_size -
                        dense_src_changed_node.shape[0],
                        dense_src_changed_node.shape[1])), dim=0)

            #new_src_ind = torch.arange(x_src.shape[0]).to(device)

            # Randomly selecting the dst (k_dst : # of nodes attached to dst)
            if 'saldst' in method:
                batch_ = torch.arange(batch_size).to(device).repeat_interleave(k_dst)
                node_ = torch.cat([torch.randperm(2*k_)[:k_.item()] for k_ in k_dst]).to(device)
                dst_idx = topk_idx_dst[batch_, node_]
            else:
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
                #dst_idx = torch.cat([torch.randperm(n).to(device)[:k_] + idx_ for n, k_ , idx_ in
                #    zip(num_nodes_list, k_dst, idx)]) # 지울 것
            #dst_idx = torch.randperm(num_nodes)[:int(num_nodes * R)]

            #Remove dst k_hop_subgraph
            remove_idx, _, _ , _ = k_hop_subgraph(node_idx=dst_idx, num_hops=k,
                                   edge_index= data.edge_index, num_nodes =num_nodes,
                                   relabel_nodes=True, ratio=ratio)
            remain = torch.ones(num_nodes).to(device).bool()
            remain[remove_idx] = False
            if manifold:
                x_dst = manifold_feature[remain]
            else:
                x_dst = data.x[remain]
            #new_src_ind = new_src_ind + len(x_dst)
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
            if 'whole' in method:
                dst_changed_node = (degree[remain] >= 0
                        ).nonzero(as_tuple=True)[0]
            dst_num_changed_node = scatter_add(degree[remain] -
                    dst_sub_degree, data.batch[remain], dim_size=batch_size)
            num_changed_node = (src_num_changed_node + dst_num_changed_node)/2
            num_changed_node = num_changed_node.long()
            dst_sub_saliency = scatter_add(saliency[remain],
                    data.batch[remain], dim_size=batch_size)
            #batch_src_changed_node = batch_src[src_changed_node]
            sum_sub_src = scatter_add(torch.ones_like(batch_src), batch_src,
                    dim_size=batch_size)
            sum_change_src = scatter_add(torch.ones_like(src_changed_node),
                    batch_src[src_changed_node], dim_size=batch_size)
            sum_sub_dst = scatter_add(torch.ones_like(batch_dst), batch_dst,
                    dim_size=batch_size)
            sum_change_dst = scatter_add(torch.ones_like(dst_changed_node),
                    batch_dst[dst_changed_node], dim_size=batch_size)
            edge_dst = dst_changed_node.repeat_interleave(sum_change_src[rand_index].repeat_interleave(sum_change_dst))

            edge_src = dense_src_changed_node[rand_index].repeat_interleave(sum_change_dst,
                    dim=0).view(-1)
            edge_src = edge_src[edge_src!=-1] + len(x_dst)

            assert len(edge_dst) == len(edge_src)

            if ('randconn' not in method) and edge_predict:
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
                #torch.stack((edge_dst, edge_src), dim=0),
                #torch.stack((edge_src, edge_dst), dim=0)
                ),
                dim=1)

            if ('randconn' not in method) and edge_predict:
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
            elif 'randconn' in method:
                edge_dst = []
                edge_src = []
                for i_b, num_cn in enumerate(num_changed_node):
                    if num_cn.item() == 0:
                        continue
                    dst_node_i = dst_changed_node[batch_dst[dst_changed_node] == i_b]
                    if len(dst_node_i) > 0:
                        dst_node_i = dst_node_i[torch.ones_like(dst_node_i).float().multinomial(num_cn.item(),
                                replacement=True)]
                    src_node_i = src_changed_node[batch_src[src_changed_node] ==
                            rand_index[i_b]]
                    if len(src_node_i) > 0:
                        src_node_i = src_node_i[torch.ones_like(src_node_i).float().multinomial(num_cn.item(),
                                replacement=True)]

                    if (len(src_node_i) > 0) and (len(dst_node_i) > 0):
                        edge_dst.append(dst_node_i)
                        edge_src.append(src_node_i + len(x_dst))
                if len(edge_dst) > 0:
                    edge_dst = torch.cat(edge_dst, dim=0)
                    edge_src = torch.cat(edge_src, dim=0)
                    edge_weight_new = torch.ones_like(edge_dst).float().repeat(2)
                    edge_weight = torch.cat((edge_weight,
                        edge_weight_new),
                        dim=0) # undirected graph
                    edge_index = torch.cat((edge_index,
                        torch.stack((edge_dst, edge_src), dim=0),
                        torch.stack((edge_src, edge_dst), dim=0)
                        ),
                        dim=1)


            new_data.x = torch.cat((x_dst, x_src), dim=0)
            if degree_feature and not manifold:
                new_degree = scatter_add(edge_weight.detach(), edge_index[0],
                        dim_size=len(new_data.x)).long()
                new_data.x = torch.zeros_like(new_data.x)
                new_data.x.scatter_(1, torch.min(new_degree[:, None],
                    torch.ones_like(new_degree[:, None]) *
                    (new_data.x.shape[1]-1)),
                        new_data.x.new_ones(len(new_data.x), 1))

            new_data.edge_index = edge_index
            new_data.edge_weight = edge_weight
            new_data.batch = torch.cat((batch_dst, new_batch_src), dim=0)

            # Size-based label
            #label_weight = k[rand_index].float() / num_nodes_list.float()

            # Saliency-based label
            #max_node = dense_saliency.shape[1]
            #mask = torch.linspace(0, max_node-1, max_node).cuda().view(1,
            #        max_node).repeat(dense_saliency.shape[0], 1) < k_src.view(dense_saliency.shape[0], 1)
            if salient_src:
                imp_src = (src_sub_saliency / saliency_batch)[rand_index] # (dense_saliency[rand_index] * mask).sum(1).float() / torch.sum(dense_saliency[rand_index], dim = -1)
                imp_dst = (dst_sub_saliency / saliency_batch)
            else: # size-based label
                imp_src = (sum_sub_src.float() /
                        num_nodes_list.float())[rand_index]
                imp_dst = sum_sub_dst.float() / num_nodes_list.float()
            #dst_imp = torch.tensor([torch.sum(imp) for imp in
            #    torch.split((saliency + 1e-10)[dst_idx], list(k_dst))]).cuda()
            #imp_dst = 1- (dst_imp / torch.sum(dense_saliency, dim = -1))
            label_weight = imp_src / (imp_src + imp_dst)

            #Augmented data training
            new_out, out_lastconv = model(data=new_data, manifold=manifold, manifold_layer=manifold_layer)
            augmented_loss = (1-label_weight) * criterion_batch(new_out, data.y.view(-1)) + label_weight * criterion_batch(new_out, data.y[rand_index].view(-1))
            augmented_loss = torch.mean(augmented_loss)
            (augmented_loss/accum_steps).backward()

            total_loss += (loss.item() + augmented_loss.item()) * num_graphs(data)
            accum += 1
            if accum == accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
                accum = 0
        #torch.cuda.empty_cache()
    if accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
    #print("edge_prediction", n_edges_correct / n_edges)
    return total_loss / len(loader.dataset)


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)[0].max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out, _ = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)

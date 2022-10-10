import time


from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch, subgraph
from utils import k_hop_subgraph
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from old_subgraph_train_eval import k_fold


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

saliency = None

import copy
from torch.utils.data import Sampler
import random

class RandomSampler(Sampler):
  def __init__(self, data_source):
    self.data_source = data_source
  def set_seed(self):
    self.seed = random.randint(0, 2**32 - 1)
  def __iter__(self):
    n = len(self.data_source)
    indexes = list(range(n))
    random.Random(self.seed).shuffle(indexes)
    return iter(indexes)
  def __len__(self):
    return len(self.data_source)


def cross_validation_with_val_set(dataset, dataset_aug_list, model, ep_net, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, method, logger=None, k=1,
                                  ratio=0.5, manifold=False, edge_predict=True,
                                  edge_thrs=0.5, only_aug=False, smoothing=0.0,
                                  train_reduce=1, proj=None):
    epochs = epochs * train_reduce
    val_accs, val_losses, accs, durations = [], [], [], []
    repeat = 3
    for _ in range(repeat):
      for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds, train_reduce))):
        dataset_ = copy.deepcopy(dataset)
        dataset_aug_list_ = copy.deepcopy(dataset_aug_list)
        if dataset.num_node_attributes > 0:
            n_attr = dataset.num_node_attributes
            mean = dataset.data.x[torch.cat((train_idx, val_idx))][:, :n_attr].mean(dim=0,
                    keepdims=True)
            std = dataset.data.x[torch.cat((train_idx, val_idx))][:, :n_attr].std(dim=0,
                    keepdims=True)
            dataset_.data.x[:, :n_attr] -= mean
            dataset_.data.x[:, :n_attr] /= std
            for dataset_aug_ in dataset_aug_list_:
                dataset_aug_.data.x[:, :n_attr] -= mean
                dataset_aug_.data.x[:, :n_attr] /= std


        train_dataset = dataset_[train_idx]
        test_dataset = dataset_[test_idx]
        val_dataset = dataset_[val_idx]
        train_aug_datasets = [dataset_aug_[train_idx] for dataset_aug_ in
                dataset_aug_list_]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            train_aug_loader = DenseLoader(train_aug_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            sampler = RandomSampler(train_dataset)
            train_loader = DataLoader(train_dataset, batch_size, shuffle=False,
                   sampler=sampler)
            train_aug_loader = [DataLoader(train_aug_dataset, batch_size,
                    shuffle=False, sampler=sampler) for train_aug_dataset in
                    train_aug_datasets]
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        ep_net.to(device)
        for m in ep_net:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        proj_param = []
        if proj is not None:
            proj.to(device)
            for m in proj:
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            proj_param = list(proj.parameters())
        optimizer = Adam(list(model.parameters()) + list(ep_net.parameters()) +
                proj_param, lr=lr, weight_decay=weight_decay)
        iters = len(train_loader)
        if lr_decay_factor < 1:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=lr_decay_factor,
                                                             #params['lr_reduce_factor'],
                                                             patience=int(1000 / iters),
                                                             #params['lr_schedule_patience'],
                                                             verbose=True)
        #if torch.cuda.is_available():
        #    torch.cuda.synchronize()

        t_start = time.perf_counter()
        stop_patience = 0
        best_acc = 0
        for epoch in range(1, epochs + 1):
            sampler.set_seed()
            if stop_patience == int(1500/iters):
                train_loss = -1
                val_losses.append(100)
                val_accs.append(-1)
                accs.append(-1)
                continue
            train_loss = train(model, ep_net, optimizer, train_loader,
                    train_aug_loader, method,
                    accum_steps=1, k=k, ratio=ratio,
                    manifold=manifold, degree_feature=dataset.degree_feature,
                    edge_predict=edge_predict, edge_thrs=edge_thrs,
                    only_aug=only_aug, smoothing=smoothing, proj=proj)
            val_losses.append(eval_loss(model, val_loader))
            val_accs.append(eval_acc(model, val_loader))
            current_val_acc = val_accs[-1]
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
            if best_acc <= current_val_acc:
                best_acc = current_val_acc
                stop_patience = 0
            else:
                stop_patience +=1

        #if torch.cuda.is_available():
        #    torch.cuda.synchronize()

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
    #loss, argmin = loss.min(dim=1)
    val_acc, argmax = val_acc.max(dim=1)
    #acc = acc[torch.arange(folds, dtype=torch.long), argmin]
    acc = acc[torch.arange(repeat * folds, dtype=torch.long), argmax]
    print(argmax)
    print(acc)
    val_acc_mean = val_acc.mean().item()
    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    #print('Val Loss: {:.4f}, Test Accuracy: {:.3f} +- {:.3f}, Duration: {:.3f}'.
    print("[Average result]")
    print('Val Acc: {:.4f}, Test Accuracy: {:.3f} +- {:.3f}, Duration: {:.3f}'.
          format(val_acc_mean, acc_mean, acc_std, duration_mean))

    #return loss_mean, acc_mean, acc_std
    return val_acc_mean, acc_mean, acc_std


#def k_fold(dataset, folds):
#    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)
#
#    test_indices, train_indices = [], []
#    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
#        test_indices.append(torch.from_numpy(idx).to(torch.long))
#
#    val_indices = [test_indices[i - 1] for i in range(folds)]
#
#    for i in range(folds):
#        train_mask = torch.ones(len(dataset), dtype=torch.bool)
#        train_mask[test_indices[i]] = 0
#        train_mask[val_indices[i]] = 0
#        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
#
#    return train_indices, test_indices, val_indices


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

def loss_cl(x1, x2):
    T = 0.5
    batch_size, _ = x1.size()

    # batch_size *= 2
    # x1, x2 = torch.cat((x1, x2), dim=0), torch.cat((x2, x1), dim=0)

    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    '''
    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    self_sim = sim_matrix[range(batch_size), list(range(int(batch_size/2), batch_size))+list(range(int(batch_size/2)))]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim - self_sim)
    loss = - torch.log(loss).mean()
    '''

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()

    return loss


def train(model, ep_net, optimizer, loader, loader_aug, method, accum_steps=1, k=1,
        ratio=0.5, manifold=False, degree_feature=False, edge_predict=True,
        edge_thrs=0.5, only_aug=False, smoothing=0.0, proj=None):
    ###GraphMIX###
    global saliency
    criterion_batch = nn.NLLLoss(reduction='none')
    model.convs[-1].register_backward_hook(backward_hook)
    model.train()

    total_loss = 0
    accum = 0
    #accum_steps = 32
    n_edges = 0
    n_edges_correct = 0
    for datas in zip(loader, *loader_aug):
      #with torch.autograd.detect_anomaly():
        data = datas[0]
        data_augs = datas[1:]
        if accum == 0:
            optimizer.zero_grad()
        data = data.to(device)
        batch_size = data.batch.max().item() + 1

        #manifold mixup
        out, out_lastconv = model(data)
        loss = label_smoothing_loss(out, data.y.view(-1), smoothing)
        total_loss += loss.item() * num_graphs(data)
        #loss.backward()
        assert torch.all(data_augs[0].y == data_augs[1].y)

        zi = proj(model(data_augs[0].to(device), cl=True))
        zj = proj(model(data_augs[1].to(device), cl=True))
        cl_loss = loss_cl(zi, zj)
        (loss + cl_loss).backward()

        #for i, data_aug in enumerate(data_augs):
        #    data_aug = data_aug.to(device)
        #    out_aug, out_lastconv_aug = model(data_aug)
        #    loss_aug = label_smoothing_loss(out_aug, data_aug.y.view(-1),
        #            smoothing) / k
        #    loss_aug.backward()
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

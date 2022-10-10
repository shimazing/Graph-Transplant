import argparse
from itertools import product

from datasets_aug import get_dataset
from gcn import GCN, GAT, GCNSkip, GIN
import torch.nn as nn

def main(args):
    if args.cl:
        from aug_train_eval import cross_validation_with_val_set
    else:
        from reuse_aug_train_eval import cross_validation_with_val_set

    layers =  [5, 4, 3]
    hiddens = [128]
    datasets = args.dataset
    lrs = [5e-4]
    ratios = [0.4, 0.2]
    nets = [GCN, GCNSkip, GIN, GAT]
    methods = args.method
    edge_predict = True
    edge_thrss = [0.5]

    results = []
    print("Manifold: ", args.manifold)
    for dataset_name, Net in product(datasets, nets):
        best_result = (-float('inf'), 0, 0)
        print('-----\n{} - {}'.format(dataset_name, Net.__name__))
        for num_layers, hidden, lr, ratio, edge_thrs in product(layers, hiddens, lrs,
                ratios, edge_thrss):
            print('------------')
            print('num_layers', num_layers, 'hidden', hidden, 'lr', lr,
                    'ratio', ratio, 'edge_thrs', edge_thrs)
            dataset_aug_list = []
            for method in methods:
                dataset, dataset_aug = get_dataset(dataset_name, sparse= True,
                        aug=method, aug_ratio=ratio)
                dataset_aug_list.append(dataset_aug)
            model = Net(dataset, num_layers, hidden)
            ep_net = nn.Sequential(nn.Linear(2 * hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden//2),
                    nn.ReLU(),
                    nn.Linear(hidden//2, 1))
            if args.cl:
                proj = {'proj': nn.Sequential(nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        )}
                dataset_aug = dataset_aug_list
                assert len(dataset_aug) == 2
            else:
                proj = dict({})
            val_acc, acc, std = cross_validation_with_val_set(
                dataset,
                dataset_aug,
                model,
                ep_net,
                folds=5,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=lr,
                lr_decay_factor=args.lr_decay_factor,
                lr_decay_step_size=args.lr_decay_step_size,
                weight_decay=0,
                method=method,
                logger=None,
                k=1,
                ratio=ratio,
                manifold=args.manifold,
                edge_predict=edge_predict,
                edge_thrs=edge_thrs,
                only_aug=args.only_aug,
                train_reduce=args.train_reduce,
                **proj
            )
            if val_acc > best_result[0]:
                best_result = (val_acc, acc, std)

        desc = '{:.3f} +- {:.3f}'.format(best_result[1], best_result[2])
        print('Best result - {}'.format(desc))
        results += ['{} - {}: {}'.format(dataset_name, model, desc)]
    print('-----\n{}'.format('\n'.join(results)))


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
        fold, epoch, val_loss, test_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=50)
    parser.add_argument('--method', type=str, default=['vanilla'],
                        help='method_type: vanilla, graphmix', nargs='+')
    parser.add_argument('--manifold', action='store_true')
    parser.add_argument('--only_aug', action='store_true')
    parser.add_argument('--dataset', type=str, default=['ENZYMES'], nargs='+')
    # none, dropN, wdropN, permE, subgraph, maskN, random4, random3, random2
    parser.add_argument('--aug_ratio', type=float, default=0.2)
    parser.add_argument('--npower', type=float, default=0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--train_reduce', type=int, default=1)
    parser.add_argument('--cl', action='store_true')

    args = parser.parse_args()
    main(args)

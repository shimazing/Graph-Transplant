import argparse
from itertools import product

from datasets import get_dataset
from gcn import GCN, GAT, GCNSkip, GIN
import torch.nn as nn

def main(args):
    if 'randconn' in args.method:
        from subgraph_train_eval import cross_validation_with_val_set
    else:
        from old_subgraph_train_eval import cross_validation_with_val_set

    layers =  args.layers
    ks = [1]
    hiddens = [128]
    datasets = args.dataset
    lrs = [5e-4]
    ratios = [1]
    nets = [GCN, GCNSkip, GAT, GIN]
    method = args.method
    edge_predict = not args.no_edgepred
    edge_thrss = [0.5]
    Rs = [0.1]

    results = []
    for dataset_name, Net in product(datasets, nets):
        best_result = (-float('inf'), 0, 0)
        print('-----\n{} - {}'.format(dataset_name, Net.__name__))
        for num_layers, hidden, lr, k, ratio, edge_thrs, R in product(layers, hiddens, lrs,
                ks, ratios, edge_thrss, Rs):
            if method == 'vanilla' and (k != 1 or ratio != 1):
                continue
            if k >= num_layers:
                continue
            print('------------')
            print('num_layers', num_layers, 'hidden', hidden, 'lr', lr, 'k', k,
                    'ratio', ratio, 'edge_thrs', edge_thrs,
                    "R", R)
            dataset = get_dataset(dataset_name, sparse= True)
            if isinstance(dataset, list):
                model = Net(dataset[0], num_layers, hidden)
            else:
                model = Net(dataset, num_layers, hidden)
            ep_net = nn.Sequential(nn.Linear(2 * hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden//2),
                    nn.ReLU(),
                    nn.Linear(hidden//2, 1))
            val_acc, acc, std = cross_validation_with_val_set(
                dataset,
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
                ratio=ratio,
                edge_predict=edge_predict,
                edge_thrs=edge_thrs,
                R=R,
                train_reduce=args.train_reduce,
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
    parser.add_argument('--dataset', type=str, nargs='+', default=['ENZYMES'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=50)
    parser.add_argument('--method', type=str, default='graphtransplant',
                        help='method_type: vanilla, graphtransplant')
    parser.add_argument('--no_edgepred', action='store_true')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--train_reduce', type=int, default=1)
    parser.add_argument('--layers', type=int, nargs='+', default=[5, 4, 3])

    args = parser.parse_args()
    main(args)

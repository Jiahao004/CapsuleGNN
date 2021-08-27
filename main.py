import os
import argparse
import logging
import random

import torch
import torch.optim as optim
from torch_geometric.data import DataLoader

from data.tudataset import preparing as tupreparing
from model import Model
from trainer import Trainer
from sklearn.model_selection import KFold
import time

seed = 12345
random.seed(seed)

def main():
    device = torch.device('cpu' if args.is_debug else 'cuda')
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    logging.basicConfig(filename=os.path.join(args.output_path, 'run.log'), level=logging.INFO)
    logging.info('''\n\n**************start at:{}*************'''.format(time.asctime(time.localtime(time.time()))))
    print(f'dataset:{args.dataset_name}')
    logging.info(f'dataset:{args.dataset_name}')
    print(f'saving to {args.output_path}')
    logging.info(f'saving to {args.output_path}')

    dataset, d_node, n_class = tupreparing(args.dataset_name, os.path.join(os.getcwd(), 'data'))

    kf = KFold(args.n_kfold, shuffle=True, random_state=seed)
    A,F,FF=[],[],[]
    for i, (train_index, test_index) in enumerate(kf.split(dataset)):
        save_path = args.output_path + '/' + str(i)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info('Fold{}:'.format(i))
        logging.info('Save to {}'.format(save_path))

        trainset = dataset[train_index.tolist()]
        testset = dataset[test_index.tolist()]

        model = Model(n_class, d_node, d_node * channel, n_gnn_layers,
                      n_prim_caps, n_digit_caps, d_node, n_caps_layers, channel, n_routing_iters,
                      is_precaps_share, dropout_p).to(device)
        optimizer = optim.Adam(model.parameters(), lr)
        trainer = Trainer(optimizer, args.patience, device)

        if args.is_debug:
            trainloader, testloader = DataLoader(trainset, args.batch_size), DataLoader(testset, args.batch_size)
            trainer.train(model, 10, testloader, testloader, save_path)
        else:
            trainloader = DataLoader(trainset, args.batch_size, shuffle=True)
            testloader = DataLoader(testset, args.batch_size, shuffle=True)
            a,f,ff=trainer.train(model, args.n_epoch, trainloader, testloader, save_path)
            A.append(a)
            F.append(f)
            FF.append(ff)

    print('done')
    logging.info('done')
    print('avg-a:{:.4f}, avg-macro-f1:{:.4f}, avg-micro-f1:{:.4f}'.format(sum(A)/len(A), sum(F)/len(F), sum(FF)/len(FF)))
    logging.info('avg-a:{:.4f}, avg-macro-f1:{:.4f}, avg-micro-f1:{:.4f}'.format(sum(A)/len(A), sum(F)/len(F), sum(FF)/len(FF)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch',
                        default=3000,
                        type=int, help='number of training epoch.')

    parser.add_argument('--batch_size',
                        default=64,
                        type=int, help='the batch size when training the epoch.')

    parser.add_argument('--dataset_name',
                        # default='ENZYMES',
                        default='MUTAG',
                        type=str, help='the name of the dataset.')

    parser.add_argument('--output_path',
                        # default='ENZYMES',
                        default='runs/MUTAG',
                        type=str, help='the output checkpoint file path', )

    parser.add_argument('--n_kfold',
                        default=5,
                        type=int, help='number of k for kfold validation')

    parser.add_argument('--patience',
                        default=100,
                        type=int or bool, help='False for no patience training')

    parser.add_argument('--is_debug',
                        default=False,
                        type=bool, help='if is true, runs on cpu, trainset and testset are the same set.')

    args = parser.parse_args()

    # define your model here
    lr = 1e-3
    channel = 2
    n_gnn_layers = 3
    n_prim_caps = 128
    n_digit_caps = 4
    n_caps_layers = 2
    n_routing_iters = 3
    is_precaps_share = True
    dropout_p = 0.1



    if args.is_debug:
        args.n_kfold = 2
    main()

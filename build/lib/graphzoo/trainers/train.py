"""
Trainer class.

    Inputs
        'lr': (0.01, 'learning rate')
        'dropout': (0.0, 'dropout probability')
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)')
        'device': ('cuda:0', 'which device to use cuda:$devicenumber for GPU or cpu for CPU')
        'epochs': (5000, 'maximum number of epochs to train for')
        'weight-decay': (0., 'l2 regularization strength')
        'momentum': (0.999, 'momentum in optimizer')
        'patience': (100, 'patience for early stopping')
        'seed': (1234, 'seed for training')
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)')
        'eval-freq': (1, 'how often to compute val metrics (in epochs)')
        'save': (0, '1 to save model and logs and 0 otherwise')
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)')
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant')
        'gamma': (0.5, 'gamma for lr scheduler')
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping')
        'min-epochs': (100, 'do not early stop before min-epochs')

"""

from __future__ import division
from __future__ import print_function
import datetime
import json
import logging
import os
import pickle
import time
import numpy as np
import torch
import torch.optim as optim
import argparse
from graphzoo.optimizers.radam import RiemannianAdam
from graphzoo.models.base_models import NCModel, LPModel
from graphzoo.utils.train_utils import get_dir_name, format_metrics
from graphzoo.dataloader.dataloader import DataLoader
# from graphzoo.data.cora.load_data import cora_download
# from graphzoo.data.pubmed.load_data import pubmed_download
# from graphzoo.data.airport.load_data import airport_download
# from graphzoo.data.disease_lp.load_data import disease_lp_download
# from graphzoo.data.disease_nc.load_data import disease_nc_download

class Trainer:
    def __init__(self,args,model, optimizer,data):

        self.args=args
        self.model=model
        self.optimizer =optimizer
        self.data=data
        self.best_test_metrics = None
        self.best_emb = None
        self.best_val_metrics = self.model.init_metric_dict()

        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        if int(self.args.cuda) >= 0:
            torch.cuda.manual_seed(self.args.seed)
    
        logging.getLogger().setLevel(logging.INFO)
        if self.args.save:
            if not self.args.save_dir:
                dt = datetime.datetime.now()
                date = f"{dt.year}_{dt.month}_{dt.day}"
                models_dir = os.path.join(os.getcwd(), self.args.dataset, self.args.task,self.args.model, date)
                self.save_dir = get_dir_name(models_dir)
            else:
                self.save_dir = self.args.save_dir
            logging.basicConfig(level=logging.INFO,
                                handlers=[
                                    logging.FileHandler(os.path.join(self.save_dir, 'log.txt')),
                                    logging.StreamHandler()
                                ])

        logging.info(f'Using: {self.args.device}')
        logging.info("Using seed {}.".format(self.args.seed))

        if not self.args.lr_reduce_freq:
            self.lr_reduce_freq = self.args.epochs

        # Model and optimizer
        logging.info(str(self.model))
    
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(self.lr_reduce_freq),
            gamma=float(self.args.gamma)
        )
        tot_params = sum([np.prod(p.size()) for p in self.model.parameters()])
        logging.info(f"Total number of parameters: {tot_params}")
        if self.args.cuda is not None and int(self.args.cuda) >= 0 :
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.cuda)
            self.model = self.model.to(self.args.device)
            for x, val in self.data.items():
                if torch.is_tensor(self.data[x]):
                    self.data[x] = self.data[x].to(self.args.device)
  
    def run(self):

        t_total = time.time()
        counter = 0
        for epoch in range(self.args.epochs):
            t = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            embeddings = self.model.encode(self.data['features'], self.data['adj_train_norm'])
            train_metrics = self.model.compute_metrics(embeddings, self.data, 'train')
            train_metrics['loss'].backward()
            if self.args.grad_clip is not None:
                max_norm = float(self.args.grad_clip)
                all_params = list(self.model.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            if (epoch + 1) % self.args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                    'lr: {}'.format(self.lr_scheduler.get_lr()[0]),
                                    format_metrics(train_metrics, 'train'),
                                    'time: {:.4f}s'.format(time.time() - t)
                                    ]))
            if (epoch + 1) % self.args.eval_freq == 0:
                self.model.eval()
                embeddings = self.model.encode(self.data['features'], self.data['adj_train_norm'])
                val_metrics = self.model.compute_metrics(embeddings, self.data, 'val')
                if (epoch + 1) % self.args.log_freq == 0:
                    logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
                if self.model.has_improved(self.best_val_metrics, val_metrics):
                    self.best_test_metrics = self.model.compute_metrics(embeddings, self.data, 'test')
                    self.best_emb = embeddings.cpu()
                    if self.args.save:
                        np.save(os.path.join(self.save_dir, 'embeddings.npy'), self.best_emb.detach().numpy())
                    self.best_val_metrics = val_metrics
                    counter = 0
                else:
                    counter += 1
                    if counter == self.args.patience and epoch > self.args.min_epochs:
                        logging.info("Early stopping")
                        break

        logging.info("Optimization Finished!")
        logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    def evaluate(self):
        if not self.best_test_metrics:
            self.model.eval()
            self.best_emb = self.model.encode(self.data['features'], self.data['adj_train_norm'])
            self.best_test_metrics = self.model.compute_metrics(self.best_emb, self.data, 'test')
        logging.info(" ".join(["Val set results:", format_metrics(self.best_val_metrics, 'val')]))
        logging.info(" ".join(["Test set results:", format_metrics(self.best_test_metrics, 'test')]))
        if self.args.save:
            np.save(os.path.join(self.save_dir, 'embeddings.npy'), self.best_emb.cpu().detach().numpy())
            if hasattr(self.model.encoder, 'att_adj'):
                filename = os.path.join(self.save_dir, self.args.dataset + '_att_adj.p')
                pickle.dump(self.model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
                print('Dumped attention adj: ' + filename)
            
            json.dump(vars(self.args), open(os.path.join(self.save_dir, 'config.json'), 'w'))
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model.pth'))
            logging.info(f"Saved model in {self.save_dir}")
        return self.best_test_metrics


# parser = argparse.ArgumentParser()
# parser.add_argument('--cuda', type=int, default=-1,
#                     help='which cuda device to use (-1 for cpu training)')
# parser.add_argument('--epochs', type=int, default=5000,
#                     help='maximum number of epochs to train for')
# parser.add_argument('--repeat', type=int, default=10,
#                     help='number of times to repeat the experiment')
# parser.add_argument('--lr', type=float, default=0.01,
#                     help='initial learning rate')
# parser.add_argument('--weight_decay', type=float, default=5e-4,
#                     help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--dropout', type=float, default=0.,
#                     help='l2 regularization strength')
# parser.add_argument('--weight-decay', type=float, default=0.2,
#                     help='dropout rate (1 - keep probability)')
# parser.add_argument('--patience', type=int, default=100,
#                     help='patience for early stopping')
# parser.add_argument('--momentum', type=float, default=0.999,
#                     help='momentum in optimizer')
# parser.add_argument('--seed', type=int, default=1234,
#                     help='seed for training')
# parser.add_argument('--log-freq', type=int, default=1, 
#                     help='how often to compute print train/val metrics (in epochs)')
# parser.add_argument('--eval-freq', type=int, default=1, 
#                     help='how often to compute val metrics (in epochs)')
# parser.add_argument('--save', type=int, default=1, 
#                     help='1 to save model and logs and 0 otherwise')
# parser.add_argument('--optimizer', type=str, default='Adam', 
#                     help='which optimizer to use, can be any of [Adam, RiemannianAdam]')
# parser.add_argument('--save-dir', type=str, default=None, 
#                     help='path to save training logs and model weights (defaults to logs/dataset/task/model/date/run/)')
# parser.add_argument('--lr-reduce-freq', type=int, default=None, 
#                     help='reduce lr every lr-reduce-freq or None to keep lr constant')
# parser.add_argument('--gamma', type=float, default=0.5,
#                     help='gamma for lr scheduler')
# parser.add_argument('--grad-clip', type=float, default=None,
#                     help='max norm for gradient clipping, or None for no gradient clipping')
# parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'disease_nc', 'disease_lp', 'airport', 'pubmed'], 
#                     help='which dataset to use')
# parser.add_argument('--model', type=str, default='GCN',
#                     choices=['Shallow', 'MLP', 'HNN', 'GCN','GAT','HGCN'], 
#                     help='which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]')
# parser.add_argument('--min-epochs', type=int, default=100,
#                     help='do not early stop before min-epochs')
# parser.add_argument('--task', type=str, default='nc',
#                     choices=['lp', 'nc'], 
#                     help='which tasks to train on, can be any of [lp, nc]')
# parser.add_argument('--dim', type=int, default=128,
#                     help='embedding dimension')
# parser.add_argument('--manifold', type=str, default='Euclidean',
#                     choices=['Euclidean', 'Hyperboloid', 'PoincareBall'], 
#                     help='which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]')
# parser.add_argument('--c', type=float, default=1.0,
#                     help='hyperbolic radius, set to None for trainable curvature')

if __name__ == '__main__':
    args = parser.parse_args()
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'

    print(args)

    # if args.dataset == 'cora':
    #     cora_download(savepath=args.datapath)

    # if args.dataset == 'disease_lp':
    #     disease_lp_download(savepath=args.datapath)

    # if args.dataset == 'disease_nc':
    #     disease_nc_download(savepath=args.datapath)

    # if args.dataset == 'airport':
    #     airport_download(savepath=args.datapath)

    # if args.dataset == 'pubmed':
    #     pubmed_download(savepath=args.datapath)

    data=DataLoader(args,args.datapath)
    result_list=[]

    for i in range(args.repeat):
        if args.task=='nc':
            model=NCModel(args)
        else:
            model=LPModel(args)

        if args.optimizer=='Adam':
            optimizer = optim.Adam(params=model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        if args.optimizer =='RiemannianAdam':
            optimizers=RiemannianAdam(params=model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

        trainer=Trainer(args,model, optimizer,data)
        trainer.run()
        result=trainer.evaluate()

        if args.task=='nc' and args.dataset in ['cora','pubmed']:
            result_list.append(100*result['acc'])

        elif args.task=='nc' and args.dataset not in ['cora','pubmed']:
            result_list.append(100*result['f1'])

        else:
            result_list.append(100*result['roc'])
            
    result_list=torch.FloatTensor(result_list)
    print("Score",torch.mean(result_list),"Error",torch.std(result_list))
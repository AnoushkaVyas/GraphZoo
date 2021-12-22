"""
Trainer class.

    Inputs
        'lr': (0.01, 'learning rate')
        'dropout': (0.0, 'dropout probability')
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)')
        'epochs': (5000, 'maximum number of epochs to train for')
        'weight-decay': (0., 'l2 regularization strength')
        'optimizer': ('Adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam]')
        'momentum': (0.999, 'momentum in optimizer')
        'patience': (100, 'patience for early stopping')
        'seed': (1234, 'seed for training')
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)')
        'eval-freq': (1, 'how often to compute val metrics (in epochs)')
        'save': (0, '1 to save model and logs and 0 otherwise')
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)')
        'sweep-c': (0, '')
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant')
        'gamma': (0.5, 'gamma for lr scheduler')
        'print-epoch': (True, '')
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
from GraphZoo.models.base_models import BaseModel
from GraphZoo.utils.train_utils import get_dir_name, format_metrics

class Trainer:
    def __init__(self,args,model, optimizer,data):

        self.args=args
        self.model=model
        self.optimizer =optimizer
        self.data=data
        self.best_test_metrics = None
        self.best_val_metrics = self.model.init_metric_dict()

        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        if int(self.args.cuda) >= 0:
            torch.cuda.manual_seed(self.args.seed)
        self.args.device = 'cuda:' + str(self.args.cuda) if int(self.args.cuda) >= 0 else 'cpu'
        self.args.patience = self.args.epochs if not self.args.patience else  int(self.args.patience)
    
        logging.getLogger().setLevel(logging.INFO)
        if self.args.save:
            if not self.args.save_dir:
                dt = datetime.datetime.now()
                date = f"{dt.year}_{dt.month}_{dt.day}"
                models_dir = os.path.join(os.environ['LOG_DIR'], self.args.task, date)
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
            self.args.lr_reduce_freq = self.args.epochs

        # Model and optimizer
        logging.info(str(self.model))
    
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(self.args.lr_reduce_freq),
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
        best_emb = None
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
                    best_emb = embeddings.cpu()
                    if self.args.save:
                        np.save(os.path.join(self.save_dir, 'embeddings.npy'), best_emb.detach().numpy())
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
            best_emb = self.model.encode(self.data['features'], self.data['adj_train_norm'])
            self.best_test_metrics = self.model.compute_metrics(best_emb, self.data, 'test')
        logging.info(" ".join(["Val set results:", format_metrics(self.best_val_metrics, 'val')]))
        logging.info(" ".join(["Test set results:", format_metrics(self.best_test_metrics, 'test')]))
        if self.args.save:
            np.save(os.path.join(self.save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
            if hasattr(self.model.encoder, 'att_adj'):
                filename = os.path.join(self.save_dir, self.args.dataset + '_att_adj.p')
                pickle.dump(self.model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
                print('Dumped attention adj: ' + filename)
            
            json.dump(vars(self.args), open(os.path.join(self.save_dir, 'config.json'), 'w'))
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model.pth'))
            logging.info(f"Saved model in {self.save_dir}")



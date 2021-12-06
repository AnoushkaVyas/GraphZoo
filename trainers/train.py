from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
import torch.optim as optim
from models.base_models import NCModel, LPModel
from models.base_models import BaseModel
from utils.train_utils import get_dir_name, format_metrics

class Trainer:
    def __init__(self,dataset: 'cora',model: BaseModel, lr: 0.01,cuda: -1,epochs: 5000, weight_decay: 0., optimizer:optim.Optimizer,
    momentum: 0.999, patience: 100, seed: 1234, log_freq : 1, eval_freq: 1, save: 0, save_dir: None,
    lr_reduce_freq: None, gamma: 0.5, print_epoch: True, grad_clip: None, min_epochs: 100,data: None):

        self.dataset=dataset
        self.lr=lr
        self.model=model
        self.cuda=cuda
        self.epochs=epochs
        self.weight_decay=weight_decay
        self.optimizer =optimizer
        self.momentum=momentum
        self.patience=patience
        self.seed=seed
        self.log_freq=log_freq
        self.eval_freq=eval_freq
        self.save=save
        self.save_dir=save_dir
        self.lr_reduce_freq=lr_reduce_freq
        self.gamma=gamma
        self.print_epoch=print_epoch
        self.grad_clip=grad_clip
        self.min_epochs=min_epochs
        self.data=data
        self.best_test_metrics = None
        self.best_val_metrics = self.model.init_metric_dict()




        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if int(self.cuda) >= 0:
            torch.cuda.manual_seed(self.seed)
        self.device = 'cuda:' + str(self.cuda) if int(self.cuda) >= 0 else 'cpu'
        self.patience = self.epochs if not self.patience else  int(self.patience)
    
        logging.getLogger().setLevel(logging.INFO)
        if self.save:
            if not save_dir:
                dt = datetime.datetime.now()
                date = f"{dt.year}_{dt.month}_{dt.day}"
                models_dir = os.path.join(os.environ['LOG_DIR'], self.task, date)
                self.save_dir = get_dir_name(models_dir)
            else:
                self.save_dir = save_dir
            logging.basicConfig(level=logging.INFO,
                                handlers=[
                                    logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                    logging.StreamHandler()
                                ])

        logging.info(f'Using: {self.device}')
        logging.info("Using seed {}.".format(self.seed))

        # Load data
        if self.task == 'nc':
            self.n_classes = int(self.data['labels'].max() + 1)
            logging.info(f'Num classes: {self.n_classes}')
        else:
            self.nb_false_edges = len(self.data['train_edges_false'])
            self.nb_edges = len(self.data['train_edges'])
            

        if not self.lr_reduce_freq:
            self.lr_reduce_freq = self.epochs

        # Model and optimizer
        logging.info(str(self.model))
    
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(self.lr_reduce_freq),
            gamma=float(self.gamma)
        )
        self.tot_params = sum([np.prod(p.size()) for p in self.model.parameters()])
        logging.info(f"Total number of parameters: {self.tot_params}")
        if self.cuda is not None and int(self.cuda) >= 0 :
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cuda)
            self.model = self.model.to(self.device)
            for x, val in self.data.items():
                if torch.is_tensor(self.data[x]):
                    self.data[x] = self.data[x].to(self.device)
        
    def run(self):

        t_total = time.time()
        counter = 0
        best_emb = None
        for epoch in range(self.epochs):
            t = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            embeddings = self.model.encode(self.data['features'], self.data['adj_train_norm'])
            train_metrics = self.model.compute_metrics(embeddings, self.data, 'train')
            train_metrics['loss'].backward()
            if self.grad_clip is not None:
                max_norm = float(self.grad_clip)
                all_params = list(self.model.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            if (epoch + 1) % self.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                    'lr: {}'.format(self.lr_scheduler.get_lr()[0]),
                                    format_metrics(train_metrics, 'train'),
                                    'time: {:.4f}s'.format(time.time() - t)
                                    ]))
            if (epoch + 1) % self.eval_freq == 0:
                self.model.eval()
                embeddings = self.model.encode(self.data['features'], self.data['adj_train_norm'])
                val_metrics = self.model.compute_metrics(embeddings, self.data, 'val')
                if (epoch + 1) % self.log_freq == 0:
                    logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
                if self.model.has_improved(self.best_val_metrics, val_metrics):
                    self.best_test_metrics = self.model.compute_metrics(embeddings, self.data, 'test')
                    best_emb = embeddings.cpu()
                    if self.save:
                        np.save(os.path.join(self.save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                    self.best_val_metrics = val_metrics
                    counter = 0
                else:
                    counter += 1
                    if counter == self.patience and epoch > self.min_epochs:
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
        if self.save:
            np.save(os.path.join(self.save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
            if hasattr(self.model.encoder, 'att_adj'):
                filename = os.path.join(self.save_dir, self.dataset + '_att_adj.p')
                pickle.dump(self.model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
                print('Dumped attention adj: ' + filename)

            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model.pth'))
            logging.info(f"Saved model in {self.save_dir}")



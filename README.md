# GraphZoo

## Introduction

GraphZoo is a training and evaluationframework. It has built-in support for several graph datasets, graph neural networks, and it can operate on different manifolds. With the unified data processing pipeline, simplified model configuration and automatic hyper-parameters tunning features equipped, GraphZoo is flexible and easy to use. The tasks supported by the framework are node classification and link prediction. 

## Installation

### Using conda

1. Install conda using this [link](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
2. git clone https://github.com/AnoushkaVyas/GraphZoo.git
3. cd GraphZoo
4. conda env create -f environment.yml
5. source set_env.sh

### Using pip

1. git clone https://github.com/AnoushkaVyas/GraphZoo.git
2. cd GraphZoo
3. virtualenv -p [PATH to python3.7 binary] GraphZoo
4. source GraphZoo/bin/activate
5. pip install -r requirements.txt
6. source set_env.sh

## Getting Started in 60 Seconds

To train a graph convolutional network model for node classification task on cora dataset:

```python
import GraphZoo as gz
import torch
from GraphZoo.config import parser

args = parser.parse_args()

data = gz.dataloader.DataLoader(args, datapath="GraphZoo/data/cora")
```

Initialize the model and fine-tune the hyperparameters:

```python
model= gz.models.NCModel(args)
```

`Trainer` is used to control the training flow:

```python
optimizer = torch.optim.Adam(model.parameters())

trainer=gz.trainers.Trainer(args,model, optimizer,data)

trainer.run()

trainer.evaluate()
```

## Customizing Input Arguments

Various flags can be modified in the `config.py` file in the source code.

### DataLoader

```
    'dataset': ('cora', 'which dataset to use')
    'val-prop': (0.05, 'proportion of validation edges for link prediction')
    'test-prop': (0.1, 'proportion of test edges for link prediction')
    'use-feats': (1, 'whether to use node features or not')
    'normalize-feats': (1, 'whether to normalize input node features')
    'normalize-adj': (1, 'whether to row-normalize the adjacency matrix')
    'split-seed': (1234, 'seed for data splits (train/test/val)')

```

### Models

```
    'task': ('nc', 'which tasks to train on, can be any of [lp, nc]')
    'model': ('GCN', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HyperGCN]')
    'dim': (128, 'embedding dimension')
    'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]')
    'c': (1.0, 'hyperbolic radius, set to None for trainable curvature')
    'r': (2., 'fermi-dirac decoder parameter for lp')
    't': (1., 'fermi-dirac decoder parameter for lp')
    'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification')
    'num-layers': (2, 'number of hidden layers in encoder')
    'bias': (1, 'whether to use bias (1) or not (0)')
    'act': ('relu', 'which activation function to use (or None for no activation)')
    'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim')
    'alpha': (0.2, 'alpha for leakyrelu in graph attention networks')
    'double-precision': ('0', 'whether to use double precision')
    'use-att': (0, 'whether to use hyperbolic attention or not')
    'local-agg': (0, 'whether to local tangent space aggregation or not')
    'n_classes': (7, 'number of classes in the dataset')
    'n_nodes': (2708, 'number of nodes in the graph') 
    'feat_dim': (1433, 'feature dimension of the dataset') 
```

### Training

```
    'lr': (0.01, 'learning rate'),
    'dropout': (0.0, 'dropout probability'),
    'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
    'epochs': (5000, 'maximum number of epochs to train for'),
    'weight-decay': (0., 'l2 regularization strength'),
    'momentum': (0.999, 'momentum in optimizer'),
    'patience': (100, 'patience for early stopping'),
    'seed': (1234, 'seed for training'),
    'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
    'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
    'save': (0, '1 to save model and logs and 0 otherwise'),
    'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
    'sweep-c': (0, ''),
    'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
    'gamma': (0.5, 'gamma for lr scheduler'),
    'print-epoch': (True, ''),
    'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
    'min-epochs': (100, 'do not early stop before min-epochs')
```
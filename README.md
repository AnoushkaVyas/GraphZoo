# GraphZoo

## Introduction

GraphZoo is a training and evaluation framework for hyperbolic graph methods. It has built-in support for several graph datasets, graph neural networks, and it can operate on different manifolds. With the unified data processing pipeline, simplified model configuration and automatic hyper-parameters tunning features equipped, GraphZoo is flexible and easy to use. The tasks supported by the framework are node classification and link prediction. 

## Installation

### From the Github source:

```

git clone https://github.com/AnoushkaVyas/GraphZoo.git
cd GraphZoo
python setup.py install
```

### From Pypi:

pip install graphzoo

## Getting Started in 60 Seconds

To train a graph convolutional network model for node classification task on cora dataset:

```python
import GraphZoo as gz
import torch
from GraphZoo.config import parser

args = parser.parse_args()

dataloader = gz.dataloader.DataLoader(args, datapath="GraphZoo/data/cora")

data=dataloader.dataloader()
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

## Customizing the Framework

### Adding Custom Dataset

1. Add the dataset files in the `data` folder of the source code.
2. To run this code on new datasets, please add corresponding data processing and loading in `load_data_nc` and `load_data_lp` functions in `dataloader/dataloader.py` in the source code.

Output format for node classification dataloader is:

```
data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
```
Output format for link prediction dataloader is:

```
data = {'adj_train': adj_train, 'features': features, ‘train_edges’: train_edges, ‘train_edges_false’: train_edges_false,  ‘val_edges’: val_edges, ‘val_edges_false’: val_edges_false, ‘test_edges’: test_edges, ‘test_edges_false’: test_edges_false, 'adj_train_norm':adj_train_norm}
```

### Adding Custom Layers

1. Attention layers can be added in `layers/att_layers.py` in the source code by adding a class in the file.
2. Hyperbolic layers can be added in `layers/hyp_layers.py` in the source code by adding a class in the file.
3. Other layers like a single GCN layer can be added in `layers/layers.py` in the source code by adding a class in the file.

### Adding Custom Models

1. After adding custom layers, custom models can be added in `models/encoders.py` in the source code by adding a class in the file.
2. After adding custom layers, custom decoders to calculate the final output can be added in `models/decoders.py` in the source code by adding a class in the file. Default decoder is the `LinearDecoder`.

## Datasets 

The included datasets are:

1. Cora
2. Pubmed
3. Disease
4. Airport

## Models In The Framework

### Shallow Methods (Shallow)
1. Shallow Euclidean
2. Shallow Hyperbolic

### Neural Network (NN) Methods
1. Multi-Layer Perceptron (MLP)
2. Hyperbolic Neural Networks (HNN) 

### Graph Neural Network (GNN) Methods
1. Graph Convolutional Neural Networks (GCN) 
2. Graph Attention Networks (GAT)
3. Hyperbolic Graph Convolutions (HGCN) 


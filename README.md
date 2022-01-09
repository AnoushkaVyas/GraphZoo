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



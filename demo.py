import graphzoo as gz
import torch
from graphzoo.config import parser

params = parser.parse_args(args=[])
params.dataset='ppi'
gz.dataloader.download_and_extract(params)
# data = gz.dataloader.DataLoader(params)


# params.task='lp'
# params.model='GCN'
# params.manifold='Euclidean'
# params.dim=128
# model= gz.models.LPModel(params)


# # optimizer = gz.optimizers.RiemannianAdam(params=model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
# optimizer = torch.optim.Adam(model.parameters())
# trainer=gz.trainers.Trainer(params,model,optimizer,data)
# trainer.run()
# trainer.evaluate()
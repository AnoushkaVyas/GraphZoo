import graphzoo as gz
import torch
from graphzoo.config import parser

args = parser.parse_args()
args.datapath='../data/cora'

data = gz.dataloader.DataLoader(args) 

model= gz.models.LPModel(args)

# optimizer = torch.optim.Adam(model.parameters())
optimizer=gz.optimizers.RiemannianAdam(params=model.parameters())

trainer=gz.trainers.Trainer(args,model, optimizer,data)

trainer.run()

trainer.evaluate()
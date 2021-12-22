# GraphZoo

```
import GraphZoo as gz
import torch
from GraphZoo.config import parser

args = parser.parse_args()

data = gz.dataloader.DataLoader(args, datapath="GraphZoo/data/cora")

model= gz.models.base_models.NCModel(args)

optimizer = gz.optimizers.RiemannianAdam(params=model.parameters())

trainer=gz.trainers.Trainer(args,model, optimizer,data)

trainer.run()

trainer.evaluate()

```
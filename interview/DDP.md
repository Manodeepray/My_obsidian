

### sources
https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html


https://docs.pytorch.org/tutorials/deep-dive.html


# DDP
In [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) (DDP) training, each rank owns a model replica and processes a batch of data, finally it uses all-reduce to sync gradients across ranks.




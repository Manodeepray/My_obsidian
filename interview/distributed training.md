ddp tutorial done
fdp tutorial 
accelerate 
tp

# DDP

[[DDP]]


# FSDP tutorial

reduces GPU memory footprint by sharding the model parameters , gradients and optimizer states

makes it feasible to train model that cannot fit on a single GPU

working principles:
- outside of backward and forward computations , the parameters are fully sharded
- before backward and forward computations , sharded params are all-gathered into unsharded params
- inside backward , local unsharded gradients are rduce scatterred to sharded-gradients


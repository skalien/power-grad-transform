# Softmax Gradient Tampering

## Link to all checkpoint files

https://drive.google.com/drive/folders/15QOyJaCETrKtbUrA6FSFPe5Nuh3SDd-j?usp=sharing

### Code related to softmax gradient tampering is located at:
https://github.com/bishshoy/softmax-gradient-tampering/blob/2560e61e5c0d241fda2e62e78cc049df3c4da9ac/timm/models/resnet.py#L831


## Table of experiments and checkpoints

| Model            | Scheduler | Hyperparameter (α) | Test Acc | Checkpoint Path                                     |
| :--------------- | :-------- | :----------------- | :------- | :-------------------------------------------------- |
| ResNet-18        | Step      | 1                  | 69.704   | resnet18/step-scheduler-alpha-1.0                   |
|                  | Step      | 0.25               | 69.844   | resnet18/step-scheduler-alpha-0.25                  |
|                  | Cosine    | 1                  | 70.208   | resnet18/cosine-scheduler-alpha-1.0                 |
|                  | Cosine    | 0.25               | 70.298   | resnet18/cosine-scheduler-alpha-0.25                |
| ResNet-18 Non-BN | Cosine    | 1                  | 66.796   | resnet_non_bn/cosine-scheduler-alpha-1.0            |
|                  | Cosine    | 0.25               | 67.796   | resnet_non_bn/cosine-scheduler-alpha-0.25           |
|                  |           |                    |          |                                                     |
| ResNet-50        | Step      | 1                  | 75.97    | resnet50/step-scheduler-alpha-1.0                   |
|                  | Step      | 0.3                | 76.494   | resnet50/step-scheduler-alpha-0.3                   |
|                  | Cosine    | 1                  | 76.56    | resnet50/cosine-scheduler-alpha-1.0                 |
|                  | Cosine    | 0.3                | 76.886   | resnet50/cosine-scheduler-alpha-0.3                 |
| ResNet-50 + LS   | Cosine    | 1                  | 76.698   | resnet50/cosine-scheduler-alpha-1.0-label-smoothing |
|                  | Cosine    | 0.3                | 76.968   | resnet50/cosine-scheduler-alpha-0.3-label-smoothing |
|                  |           |                    |          |                                                     |
| ResNet-101       | Cosine    | 1                  | 77.896   | resnet101/cosine-scheduler-alpha-1.0                |
|                  | Cosine    | 0.3                | 78.258   | resnet101/cosine-scheduler-alpha-0.3                |
|                  |           |                    |          |                                                     |
| SEResNet-18      | Cosine    | 1                  | 71.09    | seresnet18/cosine-scheduler-alpha-1.0               |
|                  | Cosine    | 0.25               | 71.436   | seresnet18/cosine-scheduler-alpha-0.25              |
|                  |           |                    |          |                                                     |
| SEResNet-50      | Cosine    | 1                  | 77.218   | seresnet50/cosine-scheduler-alpha-1.0               |
|                  | Cosine    | 0.3                | 77.952   | seresnet50/cosine-scheduler-alpha-0.3               |

Training logs can be found in the same checkpoint folder with filename `summary.csv`.

## Requirements
We recommend using the PyTorch docker container provided by NVIDIA. If you have
docker installed, you can simply use the following command to download the
PyTorch docker container.
```
docker pull nvcr.io/nvidia/pytorch:21.09-py3
```
Setup instructions are available at:
```
https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
```


## Training recipies

```
# Set NUM_GPU to number of GPUs
export NUM_GPU=4

python -m torch.distributed.run \
--standalone \
--nnodes=1 \
--nproc_per_node=$NUM_GPU \
train.py \
--data_dir IMAGENET_DIR \
--dali \
--amp \
--momentum 0.9 \
--weight-decay 5e-4 \
--sched cosine \
--model resnet18 \
--batch-size 256 \
--lr 1e-1 \
--epochs 90 \
--warmup-epochs 5 \
--cooldown-epochs 10 \
--modify-softmax 0.25 \
--logit-stats \
```

`--modify-softmax` flag effectuates softmax-gradient-tampering with the desired value of $\alpha$.

`--logit-stats` flag prints logit statistics as (norm,mean,max,var)

Possible values of `--model` can be:
```
resnet18
resnet50
resnet101
resnet18_wobn
seresnet18
seresnet101
```

Label smoothing can be invoked by:
```
--smoothing 0.1 \
```

If you want to check performance on the test set, append the following flag:
```
--validate \
```

If you want to resume from one of the aforementioned checkpoints given in the
table above, append the `--resume` flag followed by the path to the
`checkpoint.pth.tar` file:
```
--resume <path_to_checkpoint> \

# Example
--resume resnet50/cosine-scheduler-alpha-0.3-label-smoothing/model_best.pth.tar \
```

To list all arguments run:
```
python train.py -h
```

Repo is a fork of https://github.com/rwightman/pytorch-image-models.

Thanks to Ross Wightman.

# Softmax Gradient Tampering

### Link to all checkpoint files
https://drive.google.com/drive/folders/15QOyJaCETrKtbUrA6FSFPe5Nuh3SDd-j?usp=sharing

### Table of experiments and checkpoints

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


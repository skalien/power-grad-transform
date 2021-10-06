import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.d = smoothing * torch.ones(1, device='cuda')
        self.C = 1000
        print('Smoothing:', smoothing)

    def forward(self, x, target):
        logp = F.log_softmax(x, dim=-1)
        q = F.one_hot(target, num_classes=self.C)
        q_new = (1.0 - self.d) * q + self.d / self.C
        loss = (- q_new * logp).sum(dim=-1)
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class CustomLoss(nn.Module):

    class ShiftedCrossEntropyV1(nn.Module):

        def __init__(self):
            super().__init__()
            print('Loss function: Shifted Cross Entropy v1')
            self.c = 1e-3

        def forward(self, x, target):
            x = F.softmax(x, dim=-1)
            p = -(torch.log(x + self.c))
            loss = p.gather(dim=-1, index=target.unsqueeze(1)
                            ).squeeze(1).mean()
            return loss

    class ShiftedCrossEntropyV2(nn.Module):

        def __init__(self):
            super().__init__()
            print('Loss function: Shifted Cross Entropy v2')
            self.a = 0.431496
            self.c = 0.00226817

        def forward(self, x, target):
            x = F.softmax(x, dim=-1)
            p = (self.a * torch.log((x + self.c)/(1.0 + self.c)))**2
            loss = p.gather(dim=-1, index=target.unsqueeze(1)
                            ).squeeze(1).mean()
            return loss

    def __init__(self):
        super().__init__()
        self.custom_loss = self.ShiftedCrossEntropyV2()

    def forward(self, x, target):
        return self.custom_loss(x, target)

import torch


def logit_statistics(output):
    logit_norm = output.norm(dim=-1).mean()
    logit_mean = output.mean(dim=-1).abs().mean()
    logit_max = output.max(dim=-1)[0].abs().mean()
    logit_var = output.var(dim=-1).mean()

    return [
        logit_norm,
        logit_mean,
        logit_max,
        logit_var
    ]

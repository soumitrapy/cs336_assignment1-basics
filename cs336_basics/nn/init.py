import torch

def trunc_normal_(tensor, mean=0.0, std=1.0):
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-3*std, b=3*std)
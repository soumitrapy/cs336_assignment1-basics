from collections.abc import Callable, Iterable
from typing import Optional
import math

from torch.optim import Optimizer
import torch

class SGD(Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue        

                state = self.state[p]
                t = state.get('step', 0)
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)
                state['step'] = t + 1
        return loss

class AdamW(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(alpha=lr, betas=betas, eps=eps, lamda=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get('step', 0)
                alpha = group['alpha']
                beta1, beta2 = group['betas']
                eps = group['eps']
                lamda = group['lamda']

                # Update biased first moment estimate
                m = state.get('m', torch.zeros_like(p.data))
                m.mul_(beta1).add_(p.grad.data, alpha=1 - beta1)
                state['m'] = m 

                # Update biased second raw moment estimate
                v = state.get('v', torch.zeros_like(p.data))
                v.mul_(beta2).addcmul_(p.grad.data, p.grad.data, value=1 - beta2)
                state['v'] = v

                # Compute bias-corrected first and second moment estimates
                m_hat = m / (1 - beta1 ** (t + 1))
                v_hat = v / (1 - beta2 ** (t + 1))

                # U pdate parameters
                p.data -= alpha * (m_hat / (torch.sqrt(v_hat) + eps) + lamda * p.data)

                state['step'] = t + 1
        return loss

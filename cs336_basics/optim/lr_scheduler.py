import math
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer

def cosine_annealing_schedule(step: int,
                              max_lr: float,
                              min_lr: float,
                              warmup_step: int,
                              final_step: int,
                              ):
    if step < warmup_step:
        return max_lr * step / warmup_step
    elif step > final_step:
        return min_lr
    else:
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * (step - warmup_step) / (final_step - warmup_step)))

class CosineAnnealingLR(LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 min_lr: float,
                 warmup_step: int,
                 final_step: int,
                 last_epoch: int = -1,
    ):
        self.min_lr = min_lr
        self.warmup_step = warmup_step
        self.final_step = final_step
        super().__init__(optimizer = optimizer, last_epoch=last_epoch)

    def get_lr(self):
        return [cosine_annealing_schedule(step = self.last_epoch+1, max_lr = base_lr, min_lr = self.min_lr, warmup_step = self.warmup_step, final_step = self.final_step) for base_lr in self.base_lrs]
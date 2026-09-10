import math

def cosine_annealing_schedule(step: int,
                              max_lr: float,
                              min_lr: float,
                              warmup_step: int,
                              final_step: int
                              ):
    if step < warmup_step:
        return max_lr * step / warmup_step
    elif step > final_step:
        return min_lr
    else:
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * (step - warmup_step) / (final_step - warmup_step)))

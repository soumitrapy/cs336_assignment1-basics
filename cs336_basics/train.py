from fire import Fire
import numpy as np
from typing import Tuple
import os
import yaml

import torch

from cs336_basics.utils.config_utils import load_config, TrainingConfig
from cs336_basics.utils.data import get_data_loader
from cs336_basics.nn.attention import TransformerLM
from cs336_basics.optim import AdamW, CosineAnnealingLR
from cs336_basics.nn.functional import cross_entropy_loss
from cs336_basics.utils.checkpointing import save_checkpoint, load_checkpoint


def load_data(config: dict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, np.memmap, np.memmap]:
    trainds = np.memmap(config.train_path, dtype=np.uint16, mode='r')
    valds = np.memmap(config.valid_path, dtype=np.uint16, mode='r')
    traindl = get_data_loader(trainds,
                              batch_size=config.batch_size,
                              seq_len=config.context_len,
                              device=config.device,
                              )
    valdl = get_data_loader(valds,
                            batch_size=config.batch_size,
                            seq_len=config.context_len,
                            device=config.device,
                            )
    return traindl, valdl, trainds, valds

def load_model_optimizers_schedulers(config: dict) -> Tuple[TransformerLM, AdamW, CosineAnnealingLR, int]:
    iteration = 0
    model = TransformerLM(vocab_size=config.vocab_size,
                          context_len=config.context_len,
                          d_model=config.d_model,
                          d_ff=config.d_ff,
                          num_layers=config.num_layers,
                          rope_theta=config.rope_theta,
                          ).to(config.device)
    
    optimizer = AdamW(model.parameters(),
                      lr = config.lr,
                      betas = (config.beta1, config.beta2),
                      eps = config.eps,
                      weight_decay = config.weight_decay,
                      )
    scheduler = CosineAnnealingLR(optimizer,
                                  min_lr=config.min_lr,
                                  warmup_step=config.warmup_step,
                                  final_step=config.final_step,
                                  )
    if config.initial_checkpoint is not None:
        iteration = load_checkpoint(config.initial_checkpoint, model, optimizer, scheduler)
    return model, optimizer, scheduler, iteration


def train(**kwargs):
    if "config_path" in kwargs:
        config_path = kwargs.pop("config_path")
        config = load_config(config_path, kwargs)
    else:
        config = kwargs
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config = TrainingConfig(**config)
    #-------------- Data Loading --------------#
    traindl, valdl, trainds, valds = load_data(config)
    
    #-------------- Model Initialization --------------#
    model, optimizer, scheduler, iteration = load_model_optimizers_schedulers(config)

    #-------------- Training Loop --------------#
    for epoch in range(config.epochs):
        model.train()
        for batch in traindl:
            optimizer.zero_grad()
            x, y = batch
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            iteration += 1
            print(f"Epoch [{epoch+1}/{config.epochs}], Iteration [{iteration}], Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    



if __name__ == "__main__":
    Fire(train)

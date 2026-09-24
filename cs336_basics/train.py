from fire import Fire
import numpy as np
from collections.abc import Iterator, Iterable
import os
import yaml
import wandb
from tqdm import tqdm

import torch
from torch import Tensor, mode
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cs336_basics.utils.config_utils import load_config, TrainingConfig
from cs336_basics.utils.data import get_batch
from cs336_basics.nn.attention import TransformerLM
from cs336_basics.optim import AdamW, CosineAnnealingLR
from cs336_basics.nn.functional import cross_entropy_loss
from cs336_basics.utils.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.nn.utils.clip_grad import gradient_clipping

from cs336_basics.utils.seed_utils import setup_seed
from cs336_basics.utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


def load_data(config: dict) -> tuple[np.memmap, np.memmap]:
    trainds = np.memmap(config.train_path, dtype=np.uint16, mode='r')
    valds = np.memmap(config.valid_path, dtype=np.uint16, mode='r')
    return trainds, valds

def load_model_optimizers_schedulers(config: dict) -> tuple[Module, Optimizer, LRScheduler, int]:
    iteration = -1
    model = TransformerLM(vocab_size=config.vocab_size,
                          context_len=config.context_len,
                          num_layers=config.num_layers,
                          d_model=config.d_model,
                          num_heads=config.num_heads,
                          d_ff=config.d_ff,
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
  
def train_step(x: Tensor,
               y: Tensor,
               model: Module,
               optimizer: Optimizer,
               scheduler: LRScheduler,
               ) -> tuple[Tensor, Tensor]:
    model.train()
    optimizer.zero_grad()
    logits = model(x)
    loss = cross_entropy_loss(logits, y)
    loss.backward()
    grad_norm = gradient_clipping(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    return loss, grad_norm


def validation(valds: np.memmap,
               model: Module,
               config: dict,
               n_steps: int = 100,
               ) -> Tensor:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(n_steps):
            x, y = get_batch(valds, config.batch_size, config.context_len, device=config.device)
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            total_loss += loss
            total_tokens += x.numel()
    model.train()
    avg_loss = total_loss / n_steps
    return avg_loss
    

def train(**kwargs):
    if "config_path" in kwargs:
        config_path = kwargs.pop("config_path")
        config = load_config(config_path, kwargs)
    else:
        config = kwargs
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config = TrainingConfig(**config)
    setup_seed(config.seed)
    setup_logging(config)
    #-------------- Data Loading --------------#
    trainds, valds = load_data(config)
    
    #-------------- Model Initialization --------------#
    model, optimizer, scheduler, iteration = load_model_optimizers_schedulers(config)

    #-------------- Training Loop --------------#
    model.train()
    total_tokens_seen = 0
    pbar = tqdm(range(iteration+1, iteration+1+config.n_steps), desc="Training")
    for i in pbar:
        x, y = get_batch(trainds, config.batch_size, config.context_len, device=config.device)
        loss, grad_norm = train_step(x, y, model, optimizer, scheduler)
        total_tokens_seen += x.numel()

        pbar.set_postfix({
            "Iteration": i,
            "Loss": f"{loss.item():.4f}",
            "Perplexity": f"{torch.exp(loss).item():.4f}",
            "Grad Norm": f"{grad_norm.item():.4f}",
            "LR": f"{scheduler.get_last_lr()[0]:.6f}"
        })

        logger.info(f"Step {i}: Training Loss: {loss.item():.4f}, Perplexity: {torch.exp(loss).item():.4f}, Grad Norm: {grad_norm.item():.4f}, lr: {scheduler.get_last_lr()[0]:.6f}")
        wandb.log({
            "train/loss": loss.item(),
            "train/perplexity": torch.exp(loss).item(),
            "train/grad_norm": grad_norm.item(),
            "train/lr": scheduler.get_last_lr()[0],
            "train/tokens_seen": total_tokens_seen,
        },
        step=i)


        if (i+1) % config.val_interval == 0:
            val_loss = validation(valds, model, config, n_steps=config.val_steps)
            logger.info(f"Step {i}: Validation Loss: {val_loss:.4f}, perplexity: {torch.exp(val_loss).item():.4f}")
            wandb.log({
                "val/loss": val_loss.item(),
                "val/perplexity": torch.exp(val_loss).item(),
            }, step=i)


        if (i+1) % config.checkpoint_interval == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"step_{i}.pt")
            save_checkpoint(model=model, optimizer=optimizer, iteration=i, out = checkpoint_path)
            logger.info(f"Checkpoint saved at step {i} to {checkpoint_path}")
            wandb.save(checkpoint_path)

    wandb.finish()

        

if __name__ == "__main__":
    Fire(train)

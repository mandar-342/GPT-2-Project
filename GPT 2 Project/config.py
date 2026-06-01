import torch


class Config:
    block_size = 256
    batch_size = 64
    n_layers = 6
    n_heads = 8
    n_embd = 384        
    dropout = 0.3

    learning_rate = 3e-4
    min_lr = 3e-5        
    max_iters = 5000
    warmup_iters = 500    
    eval_interval = 250
    eval_iters = 100
    grad_clip = 1.0       # gradient clipping

    device = "cuda" if torch.cuda.is_available() else "cpu"

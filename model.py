from dataclasses import dataclass
import torch
import torch.nn as nn 
from torch.nn import functional as F

#---------------------------------------------------------------------------------
@dataclass 

class MHA(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_heads ==0




    


class FFN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.expand=nn.Linear(n_embd, 4* n_embd)
        self.activation=nn.GELU(approximate="tanh")
        self.recover=nn.Linear(4* n_embd, n_embd)


class decoder_layer(nn.Module):
    def __init__(sel,config):
        super().__init__()
       
        self.mha=MHA(config)
        self.ln1=nn.LayerNorm(config.n_embd)
        
        self.ffn=FFN(config)
        self.ln2=nn.LayerNorm(config.n_embd)

    def forward(x):
        x=x+ self.mha(self.ln1(x))
        x=x+ self.ffn(self.ln2(x))


class GPTConfig:
    # max number of tokens each batch 
    block_size: int=1024
    # vocabulary size
    vocab_size: int=50257
    # number of decoder layer inside decoder 
    n_layer   : int=12
    # number of heads inside MHA
    n_head    : int=12
    # embedding dimension, integers times n_heads
    n_embd     : int=768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config

        self.transformer=nn.ModuleDict(dict(
             # token embedding 
           wte =nn.Embedding(config.vocab_size,config.n_embd),
           wpe =nn.Embedding(config.block_size,config.n_embd),
             #  Block is the decoder block
            h  =nn.ModuleList([decoder_layer(Config) for _ in range(config.n_layer)]),
             # layer normalization 
           ln_f=nn.LayerNorm(config.n_embd )
          ))
        
        self.out=nn.Linear(config.n_embd,config.vocab_size, bias=False)
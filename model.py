from dataclasses import dataclass
import torch
import torch.nn as nn 
from torch.nn import functional as F
import math
import time
import torch.optim.adamw 

#---------------------------------------------------------------------------------
# We use the same name as hugging face GPT-2 for classes and paramters in order 
# to load the model as a cross check 

# The feed forward layer, which is denoted as MLP
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        # feedforward layer is taking the result from attention 
        # expand it to higher dimension to explore the richness of data structure
        # then project it back to embedding dimension
        self.c_fc=nn.Linear(config.n_embd, 4* config.n_embd)
        self.gelu=nn.GELU(approximate="tanh")
        self.c_proj=nn.Linear(4* config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1
        
    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x

# This is multi-head attention
class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head ==0
         
        # This is projection tensor fo Q, K, V
        # Notice the 3, it is the commbination of three individual projection tensors 
        self.c_attn=nn.Linear(config.n_embd,3* config.n_embd)
        # Out put projection
        self.c_proj=nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1
        # 
        self.n_head=config.n_head
        self.n_embd=config.n_embd
        # Naming convention is following OPENAI
        # This is clearly Casual mask where lower half triangle is filled with 1
        # broadcasted into batch dimension (first index) and n_heads (second index)
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size, config.block_size)
                                               .view(1,1,config.block_size, config.block_size)))

    def forward(self,x): 
        # We first get the batch size, block size( sequence length), and embdding dimension
        # These are not actual paramters in the model, not the same naming convention
        B,S,E=x.size()
        # Calculate Q,K,V
        qkv=self.c_attn(x)
        # we split the tensor in the last dimension
        # dimension of 3*n_embd into 3 dimension of n_embd
        q,k,v=qkv.split(self.n_embd,dim=2)
        # embedding dimension => n_heads * d_heads
        # switch the sequence length and the n_head in order to broadcast score calculation
        k=k.view(B,S,self.n_head,E//self.n_head ).transpose(1,2)
        q=q.view(B,S,self.n_head,E//self.n_head ).transpose(1,2)
        v=v.view(B,S,self.n_head,E//self.n_head ).transpose(1,2)
        # Calculation of the attention score
        # q dot k/ sqrt(d_heads)
        #---flash attention implementation
        # 
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        #----------------------------------
        # att= (q @ k.transpose(-2,-1)) * (1/ math.sqrt(k.size(-1)))
        # # replace the upper half of the triangle with - infinity to kill the corresponding softmax
        # att=att.masked_fill(self.bias[:,:,:S,:S]==0,float('-inf'))
        # att=F.softmax(att,dim=-1)
        # y=att @ v


        y=y.transpose(1,2).contiguous().view(B,S,E)
        y=self.c_proj(y)

        return y


# This is the decoder_layer
# In this model, we do not write a separate decoder class 
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)

    def forward(self,x):
        x=x+ self.attn(self.ln_1(x))
        x=x+ self.mlp(self.ln_2(x))
        
        return x

@dataclass
class GPTConfig:
    # max number of tokens each batch 
    block_size: int=1024
    # vocabulary size
    vocab_size: int=50257
    # number of decoder layer inside decoder 
    n_layer: int=12
    # number of heads inside MHA
    n_head: int=12
    # embedding dimension, integers times n_heads
    n_embd: int=768 
          

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config

        self.transformer=nn.ModuleDict(dict(
             # token embedding 
           wte =nn.Embedding(config.vocab_size,config.n_embd),
           wpe =nn.Embedding(config.block_size,config.n_embd),
             #  Block is the decoder block
            h  =nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
             # layer normalization 
           ln_f=nn.LayerNorm(config.n_embd)
          ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight=self.lm_head.weight

    # model initialization, need to some thinking 
    def _initial_weights(self,module):
        
        if isinstance(module,nn.Linear):
            std=0.02
            if hasattr(module,"NANOGPT_SCALE_INIT"):
                std*=(2*self.config.n_layers)** -0.5
            torch.nn.init.normal_(module.weight,mean=0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0,std=std)        



    def forward(self, idx, target=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss=None
        if target is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)), target.view(-1))

        return logits,loss
    # use classmethod to call the class from within the class
    # so we can initialize the model with the GPT-2 parameters 
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
      # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
#--------------------------------------------------------------------------------
#auto-detect the availble device 

device="cpu"
# if torch.cuda.is_available():
#     device="cuda"
# elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
#     device="mps"
# sequence_max=30
# n_sequence=5

import tiktoken

class dataloader:
    def __init__(self,B,S):
        self.B=B
        self.S=S
         
        with open('input.txt','r') as f:
            text_data=f.read() 
        enc=tiktoken.get_encoding("gpt2")
        tokens=enc.encode(text_data)
        self.tokens=torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f" 1 epoch includes {len(self.tokens)// (B* S)} batches ")
            
        self.starting_postions=0

    def next_batch(self):
            B,S= self.B,self.S
            buf= self.tokens[self.starting_postions:self.starting_postions+B*S+1]
            x= buf[:-1].view(B,S)
            y= buf[1:].view(B,S)

            self.starting_postions+=B*S
            #reset if out of bounds
            if self.starting_postions+B*S>len(self.tokens)-1:
                self.starting_position=0

            return x, y    




# text_data=text_data[1:1000]
# tokens=enc.encode(text_data)
# B,S=4,32
# buf=torch.tensor(tokens[:B*S+1],)
# x=buf[:-1].view(B,S)
# y=buf[1:].view(B,S)


model=GPT(GPTConfig(vocab_size=50304))#GPT.from_pretrained('gpt2') 
model.to(device)
#model=torch.compile(model)
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
training_set=dataloader(B=4,S=32)

for i in range(10):
    t0=time.time()
    x, y=training_set.next_batch()
    x, y=x.to(device),y.to(device)
    optimizer.zero_grad()
    logits,loss=model(x,y)
    loss.backward()
    optimizer.step()
    #torch.cuda.synchronize()
    t1=time.time()
    print(f"step:{i}, the loss is {loss.item()}, took:{t1-t0}s")





import sys; sys.exit()
# tokens=enc.encode("Hello, I'm a language model,")
# tokens=torch.tensor(tokens,dtype=torch.long)
# tokens=tokens.unsqueeze(0).repeat(n_sequence,1)
x=tokens.to(device)


torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1)< sequence_max:
    with torch.no_grad():
        logits=model(x)
        logits=logits[:,-1,:]
        prob=F.softmax(logits,dim=-1)
        topk_probs,topk_indices=torch.topk(prob,50, dim=-1)

        ix=torch.multinomial(topk_probs,1)

        xcol=torch.gather(topk_indices,-1, ix)

        x=torch.cat((x, xcol),dim=1)

for i in range(n_sequence):
    token=x[i,:sequence_max].tolist()
    decoded=enc.decode(token)
    print(">",decoded)



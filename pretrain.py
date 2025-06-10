from dataclasses import dataclass
import torch
import torch.nn as nn 
from torch.nn import functional as F
import time
import math 
import model 
from model import GPTConfig
import tiktoken
import torch.optim.adamw 
#------------------------------------------------------------- 
# auto dectection of device
device="cpu"
if torch.cuda.is_available():
    device="cuda"
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
    device="mps"
    
print(f"using device: {device}")

#-------------------------------------------------------------
# random initialization of weights
model=model.GPT(GPTConfig(vocab_size=50304))#GPT.from_pretrained('gpt2') 
model.to(device)
#model=torch.compile(model)
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),eps=1e-8)

# load the data and creat the dataloader
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
        print(f" 1 epoch includes {len(self.tokens)// (B*S)} batches ")
            
        self.starting_postions=0
    
    # get one batch for testing
    def one_batch(self):
         buf=self.tokens[:self.S+1]
         x= buf[:-1].view(1,self.S)
         y= buf[1:].view(1,self.S)
         return x, y

 
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


training_set=dataloader(B=4,S=32)

# overfit one batch
for i in range(50):
    t0=time.time()
    x, y=training_set.one_batch()
    x, y=x.to(device),y.to(device)
    optimizer.zero_grad()
    logits,loss=model(x,y)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optimizer.step()
    #torch.cuda.synchronize()
    t1=time.time()
    print(f"step:{i:4d}|loss: {loss.item()}| norm: {norm:.4f} |dt: {t1-t0}s")

from dataclasses import dataclass
import torch
import torch.nn as nn 
from torch.nn import functional as F
import math 
import model 

#------------------------------------------------------------- 
# auto dectection of device
device="cpu"
if torch.cuda.is_available():
    device="cuda"
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
    device="mps"
    
print(f"using device: {device}")

# import tiktoken
# encoder=tiktoken.get_encoding('gpt2')
# with open('input.txt','r') as f:
#     text_data=f.read()



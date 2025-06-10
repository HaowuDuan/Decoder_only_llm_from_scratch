from transformers import GPT2LMHeadModel
import torch
import torch.nn.functional as F
import tiktoken
import urllib.request
import zipfile
import os
from pathlib import Path
from torch.utils.data import Dataset, TensorDataset
import pandas as pd
## Dataloader

class data_loader(Dataset): 
    def __init__(self,batch_size, seq_length, data_path,pad_token_id=50256):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.data_path = data_path
        self.max_seq_length=self._max_seq_length()
        # Load and prepare the dataset
   
    def _load_data(self):
        data_tmp=pd.read_csv(self.data_path)
        labels= data_tmp['label'].tolist()
        content= data_tmp['content'].astype(str).tolist()
        # tokenization+ padding
        X=[]
        enc = tiktoken.get_encoding("gpt2")
        for txt in content:
            tokens=enc.encode(txt)
            tokens+=[self.pad_token_id]*(self.max_seq_length-len(tokens)) if len(tokens)<self.max_seq_length 
            X.append(tokens)
        X=torch.tensor(X,dtype=torch.long)
        Y=torch.tensor(labels,dtype=torch.long)
        return TensorDataset(X, Y)

    def _max_seq_length(self):
        # Find out the maximum sequence length in the dataset
        for 

        return 

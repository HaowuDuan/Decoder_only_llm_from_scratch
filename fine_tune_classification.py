from transformers import GPT2LMHeadModel
import model
import torch
import torch.nn.functional as F
import tiktoken
import urllib.request
import zipfile
import os
from pathlib import Path
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pandas as pd
import time
## Dataloader

class data_loader(Dataset): 
    def __init__(self,data_path,max_seq_length,pad_token_id=50256):
        self.data_path = data_path
        self.max_seq_length= max_seq_length
        self.pad_token_id = pad_token_id
        #_,_,self.max_seq_length=self._max_seq_length()
        # Load and prepare the dataset
   
    def get_dataset(self):
        data_tmp=pd.read_csv(self.data_path)
        labels= data_tmp['label'].tolist()
        content= data_tmp['content'].astype(str).tolist()
        # tokenization+ padding
        X=[]
        Y=labels
        enc = tiktoken.get_encoding("gpt2")
        for txt in content:
            tokens=enc.encode(txt)
            if len(tokens)<self.max_seq_length:
               tokens+=[self.pad_token_id]*(self.max_seq_length-len(tokens)) 
            else:
               tokens=tokens[:self.max_seq_length]
            X.append(tokens)
        X = torch.tensor(X, dtype=torch.long)
        Y = torch.tensor(Y, dtype=torch.long)
        return TensorDataset(X, Y)

train_dat=data_loader(data_path='train.csv', max_seq_length=512, pad_token_id=50256)
train_datset=train_dat.get_dataset()
dataloader = DataLoader(train_datset,batch_size=32, shuffle=False, num_workers=10)


# Model modified for classification
model=model.GPT.from_pretrained('gpt2')
# Freeze the entire model first
for param in model.parameters():
    param.requires_grad = False

for param in model.ln_f.parameters():
    param.requires_grad = True

for param in model.transformer.h[-1].parameters():
    param.requires_grad = True

torch.manual_seed(123)    
N_classes = 2
model.lm_head = torch.nn.Linear(in_features=model.config.n_embd, out_features=N_classes)

# training loop 

def train_model(model,dataloader,num_epochs,learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9,0.95),eps=1e-8)
    model.train()
    Losses=[]
    Accuracies=[]
    total_samples = len(dataloader.dataset)
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0
        N_correct=0
        
        for batch in dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs,_ = model(inputs)
            loss = F.nll_loss(outputs[:,-1,:], labels)
            loss.backward()
            optimizer.step()

            predicted = torch.max(outputs, dim=-1)
            epoch_loss += loss.item()
            N_correct += (predicted == labels).sum().item()
        

        avg_loss = epoch_loss / len(dataloader)
        accuracy = N_correct / total_samples
        Losses.append(avg_loss)
        Accuracies.append(accuracy)
        
        # Log to TensorBoard
       # writer.add_scalar('Loss/train', avg_loss, epoch)
       # writer.add_scalar('Accuracy/train', accuracy, epoch)

        Losses.append(avg_loss)
        Accuracies.append(accuracy)
        
        end_time = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}, accuracy: {N_correct/total_samples:.4f}, Time: {end_time - start_time:.2f}s') #print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}, accuracy: {N_correct/total_samples:.4f}, Time: {end_time - start_time:.2f}s')
    #writer.close()

    return Losses, Accuracies

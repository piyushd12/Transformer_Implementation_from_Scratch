import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model : int, vocab_size : int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, seq_len : int, dropout : float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Creation of matrix of shape (seq_len, d_model)
        self.PE = torch.zeros(seq_len,d_model)
        
        # Creation of vector of shape (seq_len,1)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))    
        
        # Apply sin to even and cos to odd positions    
        self.PE[:,::2] = torch.sin(position * div_term)
        self.PE[:,1::2] = torch.cos(position * div_term)

        PE = self.PE.unsqueeze(0) # Dimensions - (1, seq_len, d_model)

        self.register_buffer('PE', PE) # This will save the PE while saving the model but will not add it to the learnable parameters

    def forward(self,x):
        x = x + (self.PE[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalizaion(nn.Module):
    def __init__(self, eps : float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
    
    def forward(self,x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.gamma * ((x - mean) / (std + self.eps)) + self.beta


class FeedForwardNN(nn.Module):
    def __init__(self, d_model : int, d_ff : int, dropout : float) -> None:
        super().__init__()
        self.FFN = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff,d_model)      
        )

    def forward(self,x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.FFN(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model : int, h : int, dropout : float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model,d_model)   
        self.w_k = nn.Linear(d_model,d_model) 
        self.w_v = nn.Linear(d_model,d_model) 
        self.w_o = nn.Linear(d_model,d_model) 

    @staticmethod
    def self_attention(query,key,value,mask,dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None: 
            attention_scores.masked_fill_(mask == 0, float('-inf'))
        
        softmax = nn.Softmax(dim=-1)
        attention_scores = softmax(attention_scores)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores
        
    def forward(self,q,k,v,mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0],query.shape[1],self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttention.self_attention(query,key,value,mask,self.dropout)

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h * self.d_k)

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout : float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalizaion()

    def forward(self,x,prevLayer):
        x = x + self.dropout(prevLayer(x))
        return self.norm(x)
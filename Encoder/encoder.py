import torch
import torch.nn as nn

from utils import MultiHeadAttention, FeedForwardNN, ResidualConnection

class EncoderBlock(nn.Module):
    def __init__(self, MultiHeadAttentionBlock : MultiHeadAttention, FeedForwardNNBlock : FeedForwardNN, dropout : float) -> None:
        super().__init__()
        self.MultiHeadAttentionBlock = MultiHeadAttentionBlock
        self.FFN = FeedForwardNNBlock
        self.dropout = nn.Dropout(dropout)
        self.EncoderResidualConnections = nn.ModuleList([ResidualConnection(dropout),ResidualConnection(dropout)])

    def forward(self,x,srcMask):
        x = self.EncoderResidualConnections[0](x,lambda x : self.MultiHeadAttentionBlock(x,x,x,srcMask))
        x = self.EncoderResidualConnections[1](x,self.FFN)
        return x

class Encoder(nn.Module):
    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
    
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return x
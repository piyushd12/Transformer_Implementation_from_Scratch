import torch
import torch.nn as nn

from utils import MultiHeadAttention, FeedForwardNN, ResidualConnection

class DecoderBlock(nn.Module):
    def __init__(self, MultiHeadSelfAttentionBlock : MultiHeadAttention, CrossAttentionBlock: MultiHeadAttention, FeedForwardNNBlock : FeedForwardNN, dropout : float) -> None:
        super().__init__()
        self.MultiHeadSelfAttentionBlock = MultiHeadSelfAttentionBlock
        self.CrossAttentionBlock = CrossAttentionBlock
        self.FFN = FeedForwardNNBlock
        self.DecoderResidualConnections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self,x,encoderOutput,srcMask, tgtMask):
        x = self.DecoderResidualConnections[0](x, lambda x : self.MultiHeadSelfAttentionBlock(x,x,x,tgtMask))
        x = self.DecoderResidualConnections[1](x,lambda x : self.CrossAttentionBlock(x,encoderOutput,encoderOutput,srcMask))
        x = self.DecoderResidualConnections[2](x,self.FFN)
        return x

class Decoder(nn.Module):
    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
    
    def forward(self,x,encoderOutput, srcMask, tgtMask):
        for layer in self.layers:
            x = layer(x,encoderOutput,srcMask,tgtMask)
        return x
import torch
import torch.nn as nn

from utils import *
from Encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self,encoder : Encoder, decoder : Decoder, srcEmbed : InputEmbeddings, tgtEmbed : InputEmbeddings, srcPos : PositionalEncoding, tgtPos : PositionalEncoding, projectionLayer : ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.srcEmbed = srcEmbed
        self.tgtEmbed = tgtEmbed
        self.srcPos = srcPos
        self.tgtPos = tgtPos
        self.projectionLayer = projectionLayer

    def encode(self,src,srcMask):
        # (batch, seq_len, d_model)
        src = self.srcEmbed(src)
        src = self.srcPos(src)
        return self.encoder(src,srcMask)
    
    def decoder(self,tgt,encoderOutput, srcMask, tgtMask):
        # (batch, seq_len, d_model)
        tgt = self.tgtEmbed(tgt)
        tgt = self.tgtPos(tgt)
        return self.decoder(tgt,encoderOutput,srcMask,tgtMask)

    def project(self,x):
        # (batch, seq_len, vocab_size)
        return self.projectionLayer(x)

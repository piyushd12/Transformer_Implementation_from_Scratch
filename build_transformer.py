import torch
import torch.nn as nn

from utils import *
from Encoder import Encoder, EncoderBlock
from Decoder import Decoder, DecoderBlock
from Transformer import Transformer

def build_transformer(srcVocabSize : int, tgtVocabSize : int, srcSeqLen : int, tgtSeqLen : int, d_model : int = 512, h : int = 8, N : int = 6, dropout : float = 0.1, d_ff : int = 2048) -> Transformer:
    
    # Create input Embb
    srcEmbed = InputEmbeddings(d_model,srcVocabSize)
    tgtEmbed = InputEmbeddings(d_model,tgtVocabSize)

    # Create positional embb
    srcPos = PositionalEncoding(d_model,srcSeqLen,dropout)
    tgtPos = PositionalEncoding(d_model,tgtSeqLen,dropout)

    # Create Encoder blocks
    encoderBlocks = []
    for _ in range(N):
        encoderMultiHeadSelfAttentionBlock = MultiHeadAttention(d_model,h,dropout)
        encoderFFN = FeedForwardNN(d_model,d_ff,dropout)
        encoderBlock = EncoderBlock(encoderMultiHeadSelfAttentionBlock,encoderFFN,dropout)
        encoderBlocks.append(encoderBlock)
    
    # Create Decoder blocks
    decoderBlocks = []
    for _ in range(N):
        decoderMultiHeadSelfAttentionBlock = MultiHeadAttention(d_model,h,dropout)
        decoderCrossAttentionBlock = MultiHeadAttention(d_model,h,dropout)
        decoderFFN = FeedForwardNN(d_model,d_ff,dropout)
        decoderBlock = DecoderBlock(decoderMultiHeadSelfAttentionBlock,decoderCrossAttentionBlock,decoderFFN,dropout)
        decoderBlocks.append(decoderBlock)
    
    # Create Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoderBlocks))
    decoder = Decoder(nn.ModuleList(decoderBlocks))

    # Create output projection layer
    outputProjectionLayer = ProjectionLayer(d_model,tgtVocabSize)

    # Create Transformer
    transformer = Transformer(encoder,decoder,srcEmbed,tgtEmbed,srcPos,tgtPos,outputProjectionLayer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
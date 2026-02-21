import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class BilingualDataset(Dataset):
    def __init__(self,ds,srcTokenizer: Tokenizer,tgtTokenizer: Tokenizer,src_lang,tgt_lang,seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.srcTokenizer = srcTokenizer
        self.tgtTokenizer = tgtTokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sosToken = torch.tensor([srcTokenizer.token_to_id(['[SOS]'])],dtype=torch.int64)
        self.eosToken = torch.tensor([srcTokenizer.token_to_id(['[EOS]'])],dtype=torch.int64)
        self.padToken = torch.tensor([srcTokenizer.token_to_id(['[PAD]'])],dtype=torch.int64)
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        srcText = src_target_pair['translation'][self.src_lang]
        tgtText = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.srcTokenizer.encode(srcText,).ids
        dec_input_tokens = self.tgtTokenizer.encode(tgtText).ids

        num_enc_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        num_dec_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if num_dec_padding_tokens < 0 or num_enc_padding_tokens < 0:
            raise ValueError('Sentence is too long (padding tokens < 0)')
        
        # SOS and EOS 
        encoder_input = torch.cat(
            [
                self.sosToken,
                torch.tensor(enc_input_tokens,dtype=torch.int64),
                self.eosToken,
                torch.tensor([self.padToken] * num_enc_padding_tokens, dtype=torch.int64)
            ]
        )

        # Only SOS
        decoder_input = torch.cat(
            [
                self.sosToken,
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                torch.tensor([self.padToken] * num_dec_padding_tokens, dtype=torch.int64)
            ]
        )

        # Only EOS (target)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eosToken,
                torch.tensor([self.padToken] * num_dec_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input' : encoder_input,
            'decoder_input' : decoder_input,
            'label' : label,
            'encoder_mask' : (encoder_input != self.padToken).unsqueeze(0).unsqueeze(0),
            'decoder_mask' : (decoder_input != self.padToken).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            'src_text' : srcText,
            'tgt_text' : tgtText 
        }

def casual_mask(seq_len):
    mask = torch.triu(torch.ones(1,seq_len,seq_len),diagonal=1).type(torch.int)
    return mask == 0
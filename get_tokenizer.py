import torch
import torch.nn as nn

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config,ds,lang):
    tokenizerPath = Path(config['tokenizerFile'].format(lang))
    
    if not Path.exists(tokenizerPath):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizerPath))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizerPath))
    return tokenizer
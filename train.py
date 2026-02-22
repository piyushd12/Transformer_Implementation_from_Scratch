import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

from get_tokenizer import get_or_build_tokenizer
from build_transformer import build_transformer
from get_dataset import BilingualDataset, casual_mask
from config import get_config, get_weights_file_path

from datasets import load_dataset 

from pathlib import Path
from tqdm import tqdm
import warnings


def get_dataset(config):
    raw_ds = load_dataset("Helsinki-NLP/opus_books", f'{config['src_lang']}-{config['tgt_lang']}',split='train')

    # Get tokenizer
    srcTokenizer = get_or_build_tokenizer(config,raw_ds,config['src_lang'])
    tgtTokenizer = get_or_build_tokenizer(config,raw_ds,config['tgt_lang'])

    # Train Val split (90-10)
    trainSize = int(0.9 * len(raw_ds))
    valSize = len(raw_ds) - trainSize
    train_ds_raw, val_ds_raw = random_split(raw_ds,[trainSize,valSize])
    
    train_ds = BilingualDataset(train_ds_raw,srcTokenizer,tgtTokenizer,config['src_lang'],config['tgt_lang'],config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw,srcTokenizer,tgtTokenizer,config['src_lang'],config['tgt_lang'],config['seq_len'])

    max_src_len, max_tgt_len = 0,0

    for item in raw_ds:
        src_ids = srcTokenizer.encode(item['translation'][config['src_lang']]).ids
        tgt_ids = tgtTokenizer.encode(item['translation'][config['tgt_lang']]).ids
        max_src_len = max(max_src_len,len(src_ids))
        max_tgt_len = max(max_tgt_len,len(tgt_ids))
    
    print(f'Max Src Len -> {max_src_len} \t Max Tgt Len -> {max_tgt_len}')

    train_loader = DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
    val_loader = DataLoader(val_ds,batch_size=1,shuffle=True)

    return {
        'train_dataloader' : train_loader,
        'val_dataloader' : val_loader,
        'src_tokenizer' : srcTokenizer,
        'tgt_tokenizer' : tgtTokenizer
    }

def get_model(config, src_vocab_len, tgt_vocab_len):
    model = build_transformer(src_vocab_len,tgt_vocab_len,config['seq_len'],config['seq_len'])
    return model

def train_model(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    Path(config['checkpoint_folder']).mkdir(parents=True,exist_ok=True)

    # Define model
    ds_config = get_dataset(config)
    model = get_model(config,ds_config['src_tokenizer'].get_vocab_size(),ds_config['tgt_tokenizer'].get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(params=model.parameters(),lr=config['lr'],eps=1e-9)

    initial_epoch, global_step = 0,0
    if config['preload']:
        checkpoint_filename = get_weights_file_path(config,config['preload'])
        print(f"Preloading Model: {checkpoint_filename}")
        current_state = torch.load(checkpoint_filename)
        initial_epoch = current_state['epoch'] + 1
        optimizer.load_state_dict(current_state['optimizer_state_dice'])
        global_step = current_state['global_step']
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=ds_config['src_tokenizer'].token_to_id('[PAD]'),label_smoothing=0.1)

    for epoch in range(initial_epoch,config['num_epochs']):
        batch_iterator = tqdm(ds_config['train_dataloader'],desc=f'Processing epoch: {epoch:02d}')
        min_loss = float('inf')
        curr_loss = float('inf')
        max_score = 0
        cnt = 0
        model.train()
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # pass the tensors through transformer
            encoder_output = model.encode(encoder_input,encoder_mask)
            decoder_output = model.decode(decoder_input,encoder_output,encoder_mask,decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1,ds_config['tgt_tokenizer'].get_vocab_size()),label.view(-1))
            batch_iterator.set_postfix({'loss' : f'{loss.item():6.3f}'})

            # log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            curr_loss = loss.item()
            cnt += 1

            if cnt == 10: break
        
        validate(model,ds_config['val_dataloader'],ds_config['tgt_tokenizer'],config['seq_len'],print_msg=lambda msg: batch_iterator.write(msg),device=device)

        # Save last epoch
        print(f"saving last checkpoint. Epoch : {epoch}")
        checkpoint_filename = get_weights_file_path(config,f'{epoch:02d}')
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dice' : optimizer.state_dict(),
            'loss_score' : curr_loss,
            # 'score' : curr_score,
            'global_step' : global_step
        }, checkpoint_filename)

        # if curr_score > max_score:
        #     print(f"Saving best checkpoint. Epoch : {epoch}, loss : {curr_loss}")
        #     max_score = curr_score
        #     checkpoint_filename_best = get_weights_file_path(config,f'{epoch:02d}',isBest=True)
        #     torch.save({
        #         'epoch' : epoch,
        #         'model_state_dict' : model.state_dict(),
        #         'optimizer_state_dice' : optimizer.state_dict(),
        #         'loss_score' : curr_loss,
        #         'score' : curr_score,
        #         'global_step' : global_step
        #     }, checkpoint_filename_best)


def validate(model,val_dataloader,tgt_tokenizer,max_seq_len,print_msg,device,num_examples = 2):
    model.eval()
    print("Validation loop")

    cnt = 0
    console_width = 80 # Default

    with torch.no_grad():
        for batch in val_dataloader:
            cnt += 1
            encoder_mask = batch['encoder_mask'].to(device)
            encoder_input = batch['encoder_input'].to(device)

            assert encoder_input.size(0) == 1, "Val Batch size should be 1"
            model_out = run_encoder_once(model,encoder_input,encoder_mask,tgt_tokenizer,max_seq_len,device)
    
            src_text = batch['src_text'][0]
            tgt_text = batch['tgt_text'][0]
            model_out_text = tgt_tokenizer.decode(model_out.detach().cpu().numpy())

            print_msg('-' * console_width)
            print_msg(f"SOURCE : {src_text}")
            print_msg(f"TARGET: {tgt_text}")
            print_msg(f"PREDICTED: {model_out_text}")

            if cnt == num_examples:
                break

def run_encoder_once(model,encoder_input, enocder_mask,tokenizer,max_seq_len, device):
    sos_id = tokenizer.token_to_id('[SOS]')
    eos_id = tokenizer.token_to_id('[EOS]')

    encoder_output = model.encode(encoder_input,enocder_mask)
    decoder_input = torch.empty(1,1).fill_(sos_id).type_as(encoder_input).to(device)

    while True:
        if decoder_input.size(1) == max_seq_len: 
            break
            
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(enocder_mask).to(device)

        out = model.decode(decoder_input,encoder_output,enocder_mask,decoder_mask)

        prob = model.project(out[:,-1])

        _, next_word = torch.max(prob,dim=1)
        decoder_input = torch.cat([decoder_input,torch.empty(1,1).type_as(encoder_input).fill_(next_word.item()).to(device)],dim=1)

        if next_word == eos_id:
            break
    
    return decoder_input.squeeze(0)


if __name__ == '__main__':
    config = get_config()
    train_model(config)
    

from pathlib import Path

def get_config():
    return{
        'src_lang' : 'en',
        'tgt_lang' : 'it',
        'batch_size' : 4,
        'num_epochs' : 20,
        'seq_len' : 350,
        'd_model' : 512,
        'lr' : 10**-4,
        'checkpoint_folder' : 'weights',
        'checkpoint_basename' : 'tmodel_',
        'preload' : None,
        'tokenizer_name' : "tokenizer_{0}.json",
        'experiment_name' : 'runs/tmodel' 
    }

def get_weights_file_path(config,epoch,isBest : bool = False):
    checkpoint_folder = config['checkpoint_folder']
    checkpoint_basename = config['checkpoint_basename']
    if isBest:
        checkpoint_filename = f"{checkpoint_basename}_best.pt"
    else:
        checkpoint_filename = f"{checkpoint_basename}{epoch}_last.pt"
    # return str(Path('-') / checkpoint_folder / checkpoint_filename)
    return Path(checkpoint_folder) / checkpoint_filename
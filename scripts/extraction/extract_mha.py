import argparse
import os

import numpy as np
import torch

# Handle imports from utils
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from models.definitions.transformer_model import Transformer
from utils.data_utils import get_data_loaders, get_masks_and_count_tokens, get_src_and_trg_batches, DatasetType, LanguageDirection
from utils.constants import *
from utils.full_sentence_utils import mha_to_mha2

"""
B = batch size
S = sentence length
MD = model dimension
NH = number of heads 
Extracted input shape:  B x S x MD  
Extracted ouptut shape: B x NH x S x (MD/NH)
    
"""
def extract_input_output(training_config):
    output_path_encoder = os.path.join(training_config["output_path"], "encoder")
    output_path_decoder_self = os.path.join(training_config["output_path"], "decoder_self")
    output_path_decoder_cross = os.path.join(training_config["output_path"], "decoder_cross")
    os.makedirs(training_config['output_path'], exist_ok=True)
    os.makedirs(output_path_encoder, exist_ok=True)
    os.makedirs(output_path_decoder_self, exist_ok=True)
    os.makedirs(output_path_decoder_cross, exist_ok=True)
    
    prefix = f"{training_config['model_name']}_{training_config['dataset_name']}_{training_config['language_direction']}"
    # avoid appending to previously generated files
    for f in os.listdir(output_path_encoder):
        full_name = f"{output_path_encoder}/{f}"
        if os.path.isfile(full_name) and f.startswith(prefix):
            os.remove(full_name)
    for f in os.listdir(output_path_decoder_self):
        full_name = f"{output_path_decoder_self}/{f}"
        if os.path.isfile(full_name) and f.startswith(prefix):
            os.remove(full_name)
    for f in os.listdir(output_path_decoder_cross):
        full_name = f"{output_path_decoder_cross}/{f}"
        if os.path.isfile(full_name) and f.startswith(prefix):
            os.remove(full_name)
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!
    train_token_ids_loader, val_token_ids_loader, test_token_ids_loader, src_field_processor, trg_field_processor = get_data_loaders(
        training_config['dataset_path'],
        training_config['language_direction'],
        training_config['dataset_name'],
        training_config['batch_size'],
        device)

    pad_token_id = src_field_processor.vocab.stoi[PAD_TOKEN]  # pad token id is the same for target as well
    src_vocab_size = len(src_field_processor.vocab)
    trg_vocab_size = len(trg_field_processor.vocab)

    transformer = Transformer(
        model_dimension=BASELINE_MODEL_DIMENSION,
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        number_of_heads=BASELINE_MODEL_NUMBER_OF_HEADS,
        number_of_layers=BASELINE_MODEL_NUMBER_OF_LAYERS,
        dropout_probability=BASELINE_MODEL_DROPOUT_PROB
    ).to(device)
    checkpoint = torch.load(training_config["path_to_weights"])
    transformer.load_state_dict(checkpoint['state_dict'])
    
    mha_to_mha2(transformer, attention_type = "encoder")
    mha_to_mha2(transformer, attention_type = "decoder_self")
    mha_to_mha2(transformer, attention_type = "decoder_cross")
    
    transformer.eval()
    
    def getf(i, suffix, output_path):
        def write_input_output(model, input, output):
            # input is a tuple  (queries, keys, values, mask)
            # mask is ignored, queries, keys and values are stored separately
            # In self-attention queries = keys = values 
            q = input[0].cpu().detach().numpy()
            k = input[1].cpu().detach().numpy() 
            v = input[2].cpu().detach().numpy() 
            out = output.cpu().detach().numpy()
            in_filename_q = f"{output_path}/{prefix}_layer{i}_q_inputs_{suffix}"
            in_filename_k = f"{output_path}/{prefix}_layer{i}_k_inputs_{suffix}"
            in_filename_v = f"{output_path}/{prefix}_layer{i}_v_inputs_{suffix}"
            
            out_filename = f"{output_path}/{prefix}_layer{i}_outputs_{suffix}"
            # ad-hoc appending to the same file
            with open(in_filename_q, 'ab') as f:
                np.save(f, q)
            with open(in_filename_k, 'ab') as f:
                np.save(f, k)
            with open(in_filename_v, 'ab') as f:
                np.save(f, v)
            with open(out_filename, 'ab') as f:
                np.save(f, out)
        return write_input_output

    def extract(token_ids_loader, suffix):
        print(f"Extracting {suffix}")
        hook_handles = []
        # Register hooks on encoder self attention
        for (i, l) in enumerate(transformer.encoder.encoder_layers):
            h = l.multi_headed_attention.attention.register_forward_hook(getf(i, suffix, output_path_encoder))
            hook_handles.append(h)
        mask_filename_enc = f"{output_path_encoder}/{prefix}_masks_{suffix}"
        
        # register hooks on decoder self attention
        for (i, l) in enumerate(transformer.decoder.decoder_layers):
            h = l.trg_multi_headed_attention.attention.register_forward_hook(getf(i, suffix, output_path_decoder_self))
            hook_handles.append(h)
        mask_filename_dec_self = f"{output_path_decoder_self}/{prefix}_masks_{suffix}"
        
        # register hooks on decoder cross attention
        for (i, l) in enumerate(transformer.decoder.decoder_layers):
            h = l.src_multi_headed_attention.attention.register_forward_hook(getf(i, suffix, output_path_decoder_cross))
            hook_handles.append(h)
        mask_filename_dec_cross = f"{output_path_decoder_cross}/{prefix}_masks_{suffix}"
        mask_filename_dec_cross_src = f"{output_path_decoder_cross}/{prefix}_masks_{suffix}_src"
        
        
        for batch_idx, token_ids_batch in enumerate(token_ids_loader):
            if (batch_idx % training_config['console_log_freq'] == 0):
                print(f"Current batch in {suffix}: {batch_idx}")
                
            src_token_ids_batch, trg_token_ids_batch_input, _ = get_src_and_trg_batches(token_ids_batch)
            src_mask, trg_mask, num_src_tokens, num_trg_tokens = get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch_input, pad_token_id, device)
            

            with open(mask_filename_enc, 'ab') as f:
                np.save(f, src_mask.cpu().detach().numpy())
                
            with open(mask_filename_dec_self, 'ab') as f:
                np.save(f, trg_mask.cpu().detach().numpy())
            
            with open(mask_filename_dec_cross, 'ab') as f:
                np.save(f, trg_mask.cpu().detach().numpy())
            
            with open(mask_filename_dec_cross_src, 'ab') as f:
                np.save(f, src_mask.cpu().detach().numpy())
            
            
            transformer.forward(src_token_ids_batch, trg_token_ids_batch_input, src_mask, trg_mask)

        for h in hook_handles:
            h.remove()
            
    extract(val_token_ids_loader, "val")
    extract(train_token_ids_loader, "train")
    extract(test_token_ids_loader, "test")

if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    num_warmup_steps = 4000

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="target number of tokens in a src/trg batch", default=1500)

    # Data related args
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='which dataset to use for training', default=DatasetType.IWSLT.name)
    parser.add_argument("--language_direction", choices=[el.name for el in LanguageDirection], help='which direction to translate', default=LanguageDirection.E2G.name)
    parser.add_argument("--dataset_path", type=str, help='download dataset to this path', default=DATA_DIR_PATH)

    # Logging/debugging related (helps a lot with experimentation)
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq", default=10)
    parser.add_argument("--model_name", type=str, help="name of the model", default = ' 128emb_20ep')
    parser.add_argument("--path_to_weights", type=str, help="path to the weights to load", required=True)
    parser.add_argument("--output_path", type = str, help = "path where the extracted values should be saved", default = MHA_OUTPUT_PATH)
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    training_config['num_warmup_steps'] = num_warmup_steps
    output_path_encoder = os.path.join(training_config["output_path"], "encoder")
    output_path_decoder_self = os.path.join(training_config["output_path"], "decoder_self")
    output_path_decoder_cross = os.path.join(training_config["output_path"], "decoder_cross")    
    extract_input_output(training_config)

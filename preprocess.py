"""
    Notes:
        * I won't add model checkpoint averaging as mentioned in the paper - it just feels like an arbitrary heuristic
         and it won't add anything to the learning experience this repo aims to provide.

"""


import argparse
import time
import os

import torch
from torch import nn
from torch.optim import Adam
import numpy as np


from models.definitions.transformer_model import Transformer
from utils.data_utils import get_datasets_and_vocabs, get_masks_and_count_tokens_src, DatasetType, LanguageDirection
from utils.constants import *
from utils.visualization_utils import visualize_attention
from utils.decoding_utils import greedy_decoding, get_beam_decoder, DecodingMethod
from utils.utils import print_model_metadata
from utils.resource_downloader import download_models

from torch.utils.tensorboard import SummaryWriter


from utils.optimizers_and_distributions import CustomLRAdamOptimizer, LabelSmoothingDistribution
from models.definitions.transformer_model import Transformer
from utils.data_utils import get_data_loaders, get_masks_and_count_tokens, get_src_and_trg_batches, DatasetType, LanguageDirection
import utils.utils as utils
from utils.constants import *


# Global vars for logging purposes
num_of_trg_tokens_processed = 0
bleu_scores = []
global_train_step, global_val_step = [0, 0]
writer = SummaryWriter()  # (tensorboard) writer will output to ./runs/ directory by default
PREPROCESS_DIR_PATH = os.environ["SCRATCH"]

# Simple decorator function so that I don't have to pass these arguments every time I call get_train_val_loop
def get_eval_loop(baseline_transformer, label_smoothing, pad_token_id, time_start):

    def eval_loop(token_ids_loader, mask_output_path):
    
        global num_of_trg_tokens_processed, global_train_step, global_val_step, writer
        device = next(baseline_transformer.parameters()).device
        src_mask_accumulator = None
        output_batch_count = 0
        #
        # Main loop - start of the CORE PART
        #
        for batch_idx, token_ids_batch in enumerate(token_ids_loader):
            src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt = get_src_and_trg_batches(token_ids_batch)
            src_mask, trg_mask, num_src_tokens, num_trg_tokens = get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch_input, pad_token_id, device)
            # log because the KL loss expects log probabilities (just an implementation detail)
#            predicted_log_distributions = baseline_transformer(src_token_ids_batch, trg_token_ids_batch_input, src_mask, trg_mask)
#            smooth_target_distributions = label_smoothing(trg_token_ids_batch_gt)  # these are regular probabilities
            
            # Save masks
            src_mask = src_mask.cpu()
            src_mask_accumulator = src_mask if src_mask_accumulator is None else torch.cat([src_mask_accumulator, src_mask], dim = 0)
            if src_mask_accumulator.shape[0] >= FLUSH_SIZE:
                print(src_mask_accumulator.shape)
                print(src_mask_accumulator.squeeze().numpy().shape)
                np.save(os.path.join(mask_output_path, f"mask-batch-{output_batch_count}"), src_mask_accumulator.squeeze().numpy())
                src_mask_accumulator = None
                output_batch_count += 1

            if batch_idx % 100 == 0:
                print(f"batch: {batch_idx}")
    return eval_loop

    
def train_transformer(preprocess_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!
    # Step 1: Prepare data loaders
    train_token_ids_loader, val_token_ids_loader, src_field_processor, trg_field_processor = get_data_loaders(
        preprocess_config['dataset_path'],
        preprocess_config['language_direction'],
        preprocess_config['dataset_name'],
        preprocess_config['batch_size'],
        device,
        fix_length=MAX_LEN,
        batch_size_tokens = False)

    pad_token_id = src_field_processor.vocab.stoi[PAD_TOKEN]  # pad token id is the same for target as well
    src_vocab_size = len(src_field_processor.vocab)
    trg_vocab_size = len(trg_field_processor.vocab)

    # Step 2: Prepare the model (original transformer) and push to GPU
    baseline_transformer = Transformer(
        model_dimension=BASELINE_MODEL_DIMENSION,
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        number_of_heads=BASELINE_MODEL_NUMBER_OF_HEADS,
        number_of_layers=BASELINE_MODEL_NUMBER_OF_LAYERS,
        dropout_probability=BASELINE_MODEL_DROPOUT_PROB
    ).to(device)

    model_path = os.path.join(BINARIES_PATH, preprocess_config['model_name'])
    if not os.path.exists(model_path):
        print(f'Model {model_path} does not exist, attempting to download.')
        model_path = download_models(preprocess_config)

    model_state = torch.load(model_path)
    print_model_metadata(model_state)

    baseline_transformer.load_state_dict(model_state["state_dict"], strict=True)
    baseline_transformer.eval()
      
    # Makes smooth target distributions as opposed to conventional one-hot distributions
    # My feeling is that this is a really dummy and arbitrary heuristic but time will tell.
    label_smoothing = LabelSmoothingDistribution(BASELINE_MODEL_LABEL_SMOOTHING_VALUE, pad_token_id, trg_vocab_size, device)

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    train_val_loop = get_eval_loop(baseline_transformer, label_smoothing, pad_token_id, time.time())

    # Step 4: Start the training
    with torch.no_grad():
        # Set parameters to save intermediate results for test set
        for layer_index, l in enumerate(baseline_transformer.encoder.encoder_layers):
            dir_path =  os.path.join(PREPROCESS_DIR_PATH, "encoder","train", f"l{layer_index}")
            os.makedirs(dir_path, exist_ok = True)
            l.set_save_intermediate(layer_index, dir_path)
        mask_path = os.path.join(PREPROCESS_DIR_PATH, "encoder","train","src_mask")
        os.makedirs(mask_path, exist_ok = True)
        train_val_loop(token_ids_loader=train_token_ids_loader, mask_output_path = mask_path)
        
        # Set parameters to save intermediate results for test set
        for layer_index, l in enumerate(baseline_transformer.encoder.encoder_layers):
            dir_path =  os.path.join(PREPROCESS_DIR_PATH, "encoder", "val", f"l{layer_index}")
            os.makedirs(dir_path, exist_ok = True)
            l.set_save_intermediate(layer_index, dir_path)
        mask_path = os.path.join(PREPROCESS_DIR_PATH, "encoder","val","src_mask")
        os.makedirs(mask_path, exist_ok = True)
        train_val_loop(token_ids_loader=val_token_ids_loader, mask_output_path = mask_path)

      




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="target number of tokens in a src/trg batch", default=50)
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='which dataset to use for training', default=DatasetType.IWSLT.name)
    parser.add_argument("--language_direction", choices=[el.name for el in LanguageDirection], help='which direction to translate', default=LanguageDirection.E2G.name)
    parser.add_argument("--dataset_path", type=str, help='download dataset to this path', default=DATA_DIR_PATH)
    parser.add_argument("--model_name", type=str, help="transformer model name", default=r'transformer_128.pth')
    parser.add_argument("--part", type=str, help="transformer model name", default=r'transformer_128.pth')

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    preprocess_config = dict()
    for arg in vars(args):
        preprocess_config[arg] = getattr(args, arg)

    # Train the original transformer model
    train_transformer(preprocess_config)

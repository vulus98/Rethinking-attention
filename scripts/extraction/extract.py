import argparse
import os

import numpy as np

import torch

# Local imports
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from models.definitions.transformer_model import Transformer
from utils.data_utils import get_data_loaders, get_masks_and_count_tokens_src, get_src_and_trg_batches, DatasetType, LanguageDirection
from utils.constants import *
from utils.simulator import restructure_encoder_layers

def extract_input_output(training_config):
    prefix = f"{training_config['model_name']}_{training_config['dataset_name']}_{training_config['language_direction']}"
    # avoid appending to previously generated files
    for f in os.listdir(LAYER_OUTPUT_PATH):
        full_name = f"{LAYER_OUTPUT_PATH}/{f}"
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
    restructure_encoder_layers(transformer)

    transformer.eval()

    train_token_ids_loader, val_token_ids_loader, test_token_ids_loader, src_field_processor, trg_field_processor = get_data_loaders(
        training_config['dataset_path'],
        training_config['language_direction'],
        training_config['dataset_name'],
        training_config['batch_size'],
        device,
        max_len_train=MAX_LEN)
    
    def getf(i, extra_pref, suffix):
        def write_input_output(model, input, output):
            # input is a tuple with the embeddings in first place
            inp = input[0].cpu().detach().numpy()
            out = output.cpu().detach().numpy()
            in_filename = f"{LAYER_OUTPUT_PATH}/{prefix}_{extra_pref}_layer{i}_inputs_{suffix}"
            out_filename = f"{LAYER_OUTPUT_PATH}/{prefix}_{extra_pref}_layer{i}_outputs_{suffix}"
            # ad-hoc appending to the same file
            with open(in_filename, 'ab') as f:
                np.save(f, inp)
            with open(out_filename, 'ab') as f:
                np.save(f, out)
        return write_input_output

    def extract(token_ids_loader, suffix):
        print(f"Extracting {suffix}")
        hook_handles = []
        for (i, l) in enumerate(transformer.encoder.encoder_layers):
            h_ELR = l.register_forward_hook(getf(i, "ELR", suffix))
            h_ALR = l.sublayer_zero.layer.register_forward_hook(getf(i, "ALR", suffix))
            h_ALRR = l.sublayer_zero.register_forward_hook(getf(i, "ALRR", suffix))
            hook_handles.append(h_ELR)
            hook_handles.append(h_ALR)
            hook_handles.append(h_ALRR)
        hook_handles.append(transformer.encoder.norm.register_forward_hook(getf("norm", "ELR", suffix)))
        mask_filename = f"{LAYER_OUTPUT_PATH}/{prefix}_masks_{suffix}"

        for batch_idx, token_ids_batch in enumerate(token_ids_loader):
            if (batch_idx % training_config['console_log_freq'] == 0):
                print(f"Current batch in {suffix}: {batch_idx}")
            src_token_ids_batch, _, _ = get_src_and_trg_batches(token_ids_batch)
            src_mask, num_src_tokens = get_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id)
            with open(mask_filename, 'ab') as f:
                np.save(f, src_mask.cpu().detach().numpy())
            transformer.encode(src_token_ids_batch, src_mask)

        for h in hook_handles:
            h.remove()
    
    extract(train_token_ids_loader, "train")
    extract(val_token_ids_loader, "val")
    extract(test_token_ids_loader, "test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="target number of tokens in a src/trg batch", default=1500)

    # Data related args
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='which dataset to use for training', default=DatasetType.IWSLT.name)
    parser.add_argument("--language_direction", choices=[el.name for el in LanguageDirection], help='which direction to translate', default=LanguageDirection.G2E.name)
    parser.add_argument("--dataset_path", type=str, help='download dataset to this path', default=DATA_DIR_PATH)

    # Logging/debugging related (helps a lot with experimentation)
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq", default=10)
    parser.add_argument("--model_name", type=str, help="name of the model", default = "128emb_20ep")
    parser.add_argument("--path_to_weights", type=str, help="path to the weights of the trained transformer", required=True)
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    extract_input_output(training_config)

import argparse
import time
import pickle


import torch
from torch import nn
from torch.optim import Adam
from tensorboardX import SummaryWriter


from utils.optimizers_and_distributions import CustomLRAdamOptimizer, LabelSmoothingDistribution
from models.definitions.transformer_model import Transformer
from utils.data_utils import get_data_loaders, get_masks_and_count_tokens, get_src_and_trg_batches, DatasetType, LanguageDirection
import utils.utils as utils
from utils.constants import *

def extract_input_output(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    train_token_ids_loader, val_token_ids_loader, src_field_processor, trg_field_processor = get_data_loaders(
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

    transformer.eval()

    def getf(i, is_train=True):
        def write_input_output(model, input, output):
            inp = input[0].detach()
            out = output.detach()
            in_filename = f"{LAYER_OUTPUT_PATH}/{training_config['model_name']}_layer{i}_inputs_{'train' if is_train else 'val'}"
            out_filename = f"{LAYER_OUTPUT_PATH}/{training_config['model_name']}_layer{i}_outputs_{'train' if is_train else 'val'}"
            with open(in_filename, 'ab') as f:
                pickle.dump(inp, f)
            with open(out_filename, 'ab') as f:
                pickle.dump(out, f)
        return write_input_output

    # extract train dataset activations
    hook_handles = []
    for (i, l) in enumerate(transformer.encoder.encoder_layers):
        h = l.sublayers[0].register_forward_hook(getf(i, True))
        hook_handles.append(h)

    for batch_idx, token_ids_batch in enumerate(train_token_ids_loader):
        src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt = get_src_and_trg_batches(token_ids_batch)
        src_mask, trg_mask, num_src_tokens, num_trg_tokens = get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch_input, pad_token_id, device)
        transformer.encode(src_token_ids_batch, src_mask)

    for h in hook_handles:
        h.remove()

    # extract validation dataset activations
    hook_handles = []
    for (i, l) in enumerate(transformer.encoder.encoder_layers):
        h = l.sublayers[0].register_forward_hook(getf(i, False))
        hook_handles.append(h)

    for batch_idx, token_ids_batch in enumerate(val_token_ids_loader):
        src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt = get_src_and_trg_batches(token_ids_batch)
        src_mask, trg_mask, num_src_tokens, num_trg_tokens = get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch_input, pad_token_id, device)
        transformer.encoder(src_token_ids_batch, src_mask)

if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    num_warmup_steps = 4000

    #
    # Modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    # You should adjust this for your particular machine (I have RTX 2080 with 8 GBs of VRAM so 1500 fits nicely!)
    parser.add_argument("--batch_size", type=int, help="target number of tokens in a src/trg batch", default=1500)

    # Data related args
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='which dataset to use for training', default=DatasetType.IWSLT.name)
    parser.add_argument("--language_direction", choices=[el.name for el in LanguageDirection], help='which direction to translate', default=LanguageDirection.E2G.name)
    parser.add_argument("--dataset_path", type=str, help='download dataset to this path', default=DATA_DIR_PATH)

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq", default=10)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq", default=1)
    parser.add_argument("--model_name", type=str, help="name of the model", default="")
    parser.add_argument("--path_to_weights", type=str, help="path to the weights to load", required=True)
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    training_config['num_warmup_steps'] = num_warmup_steps

    # Train the original transformer model
    extract_input_output(training_config)

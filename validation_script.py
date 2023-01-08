"""
    Notes:
        * I won't add model checkpoint averaging as mentioned in the paper - it just feels like an arbitrary heuristic
         and it won't add anything to the learning experience this repo aims to provide.

"""


import argparse
import time


import torch
from tensorboardX import SummaryWriter
from training_FF import FFNetwork
from models.definitions.transformer_model import Transformer, replace_sublayer
from models.definitions.transformer_model import Transformer
from utils.data_utils import get_data_loaders, get_masks_and_count_tokens, get_src_and_trg_batches, DatasetType, LanguageDirection
import utils.utils as utils
from utils.constants import *





def evaluate_transformer(evaluate_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!
    # Step 1: Prepare data loaders
    train_token_ids_loader, val_token_ids_loader, test_token_ids_loader, src_field_processor, trg_field_processor = get_data_loaders(
        evaluate_config['dataset_path'],
        evaluate_config['language_direction'],
        evaluate_config['dataset_name'],
        evaluate_config['batch_size'],
        device)

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
    model_path = os.path.join(BINARIES_PATH, evaluate_config['model_name'])
    model_state = torch.load(model_path)
    baseline_transformer.load_state_dict(model_state["state_dict"], strict=True)
    baseline_transformer.eval()
    
    # Step 3: Substitute attention layers
    for i in range(0):
        FF_net = FFNetwork()
        baseline_transformer = replace_sublayer(baseline_transformer, FF_net, i, device = device)
    
    # Step 4: Compute BLEU
    utils.calculate_bleu_score(baseline_transformer, val_token_ids_loader, trg_field_processor)

if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    num_warmup_steps = 4000
    # Modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="transformer model name", default=r'transformer_128.pth')

    # Keep these 2 in sync with the model you pick via model_name
    parser.add_argument("--dataset_name", type=str, choices=['IWSLT', 'WMT14'], help='which dataset to use for training', default=DatasetType.IWSLT.name)
    parser.add_argument("--language_direction", type=str, choices=[el.name for el in LanguageDirection], help='which direction to translate', default=LanguageDirection.E2G.name)

    # Cache files and datasets are downloaded here during training, keep them in sync for speed
    parser.add_argument("--dataset_path", type=str, help='download dataset to this path', default=DATA_DIR_PATH)
    parser.add_argument("--batch_size", type=int, help="target number of tokens in a src/trg batch", default=1500)

    # Decoding related args
    args = parser.parse_args()
    # Wrapping training configuration into a dictionary
    evaluate_config = dict()
    for arg in vars(args):
        evaluate_config[arg] = getattr(args, arg)
    evaluate_config['num_warmup_steps'] = num_warmup_steps

    # Train the original transformer model
    evaluate_transformer(evaluate_config)
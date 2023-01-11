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
from models.definitions.transformer_model import Transformer, mha_to_mha2, replace_mha, replace_sublayer
from models.definitions.transformer_model import Transformer
from utils.data_utils import get_data_loaders, get_masks_and_count_tokens, get_src_and_trg_batches, DatasetType, LanguageDirection
import utils.utils as utils
from utils.constants import *
from training_mh_separate_heads import FFNetwork_small
from utils.constants import *

devices=list(range(torch.cuda.device_count()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!
# device = "cpu"
test = True
def substitute_mha_only(baseline_transformer, substitute_class, substitute_model_path, layers, epoch):
    import models.definitions.mha_only_FF as m
    FF_net = getattr(m, substitute_class)
    print(f"Substituing attention with {FF_net}")
    mha_to_mha2(baseline_transformer)
    layers = [int(layers)] if layers is not None else range(6)
    print(layers)
    for l in layers:
        ff_net = FF_net().to(device)
        if not test:
            model_path=os.path.join(substitute_model_path, f'l{l}', MHA_ONLY_CHECKPOINT_FORMAT.format(epoch, l))
            print(f"Loading weights from {model_path}")
            model_state = torch.load(model_path)
            ff_net.load_state_dict(model_state)
        else:
            print("Test uninitialized")
        ff_net.eval()
        replace_mha(baseline_transformer, ff_net, l, device)
    
def substitute_mha(baseline_transformer, substitute_class, substitute_model_path, layers, epoch):
    import models.definitions.mha_FF as m #TODO: place you FF_net definitions in this file
    FF_net = getattr(m, substitute_class)
    print(f"Substituing attention with {FF_net}")
    mha_to_mha2(baseline_transformer)
    layers = [int(layers)] if layers is not None else range(6)
    print(layers)
    for l in layers:
        ff_net = FF_net()
        if not test:
            model_path=os.path.join(substitute_model_path, f'l{l}', MHA__CHECKPOINT_FORMAT.format(epoch, l)) # TODO: modify according to your naming
            model_state = torch.load(model_path)
            ff_net.load_state_dict(model_state)
        ff_net.eval()
        replace_mha(baseline_transformer, ff_net, l, device)
    

def substitute_attention(baseline_transformer, substitute_class, substitute_model_path, layer, epoch, t):
    if t == "mha_only":
        print("Substitute mha only")
        substitute_mha_only(baseline_transformer, substitute_class, substitute_model_path, layer, epoch)
    if t == "mha":
        print("Substitute mha layer")
        substitute_mha(baseline_transformer, substitute_class, substitute_model_path, layer, epoch)

def evaluate_transformer(evaluate_config):
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
    # Step 3: substitute attention
    if evaluate_config["substitute_type"] != "none":
        substitute_attention(baseline_transformer, 
                             evaluate_config["substitute_class"], 
                             evaluate_config["substitute_model_path"], 
                             evaluate_config["layer"],
                             evaluate_config["epoch"],
                             evaluate_config["substitute_type"]) 
    else:
        print("#"*100)
        print("\n\t NO SUBSTITUTION \n")
        print("#"*100)
        
    # Step 4: Compute BLEU
    with torch.no_grad():
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
    parser.add_argument("--substitute_class", type=str, help="class that substitutes attention e.g. FF_large")
    parser.add_argument("--substitute_model_path", type=str, help="path to the substitue of attention. The folder should contain 6 subfolders one for each layer. Inside the FF checkpoints are stored with name: ff_network_{epoch}_layer_{layer}.pth")
    parser.add_argument("--layer", help = "If layer is not specified, all layers are substituted", default = None)
    parser.add_argument("--epoch", type = int, help="Epoch checkpoint to use.")
    parser.add_argument("--substitute_type", type = str, help="Epoch checkpoint to use.", choices=["mha", "mha_only", "mha_separate_heads", "none"], default="none")
    
    # Decoding related args
    args = parser.parse_args()
    # Wrapping training configuration into a dictionary
    evaluate_config = dict()
    for arg in vars(args):
        evaluate_config[arg] = getattr(args, arg)
    print(evaluate_config)
    evaluate_config['num_warmup_steps'] = num_warmup_steps

    # Train the original transformer model
    evaluate_transformer(evaluate_config)
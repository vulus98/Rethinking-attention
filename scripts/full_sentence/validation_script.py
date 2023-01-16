import argparse

import torch


# Local imports
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from models.definitions.transformer_model import Transformer
from utils.data_utils import get_data_loaders, DatasetType, LanguageDirection
from utils.full_sentence_utils import substitute_attention
import utils.utils as utils
from utils.constants import *

devices=list(range(torch.cuda.device_count()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

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
                             evaluate_config["layers"],
                             evaluate_config["epoch"],
                             evaluate_config["substitute_type"],
                             untrained = evaluate_config["untrained"]) 
    else:
        print("#"*100)
        print("\n\t NO SUBSTITUTION IN ENCODER\n")
        print("#"*100)
        
    if evaluate_config["substitute_type_d"] != "none":
        substitute_attention(baseline_transformer, 
                             evaluate_config["substitute_class_d"], 
                             evaluate_config["substitute_model_path_d"], 
                             evaluate_config["layers_d"],
                             evaluate_config["epoch_d"],
                             evaluate_config["substitute_type_d"],
                             untrained = evaluate_config["untrained"],
                             multi_device = evaluate_config["multi_device_d"], decoder = True)
    else:
        print("#"*100)
        print("\n\t NO SUBSTITUTION IN DECODER\n")
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
    parser.add_argument("--multi_device_d", action = "store_true", help =  "Ignore, useful for bigger models which are not used")
    parser.add_argument("--multi_device", action = "store_true", help =  "Ignore, useful for bigger models which are not used")
    
    # Params for encoder substitution
    parser.add_argument("--substitute_class", type=str, help="class that substitutes attention e.g. FFNetwork_shrink", default = "None")
    parser.add_argument("--substitute_model_path", type=str, help="path to the substitue of attention. The folder should contain 6 subfolders one for each layer. Inside the FF checkpoints are stored with name: ff_network_{epoch}_layer_{layer}.pth", default = None)
    parser.add_argument("--layers", nargs='+',type = int ,help = "List of layers to substitute. If layer is not specified, all layers are substituted")
    parser.add_argument("--epoch", type = int, help="Epoch checkpoint to use.")
    parser.add_argument("--untrained", action = "store_true")
    parser.add_argument("--substitute_type", type = str, help="Type of approach to use for substitution", choices=["ALRR", "mha_only", "mha_separate_heads", "none"], default="none")
    
    # Params for decoder substitution
    parser.add_argument("--substitute_class_d", type=str, help="class that substitutes attention e.g. FFNetwork_shrink", default="None")
    parser.add_argument("--layers_d", nargs='+',type = int ,help = "List of layers to substitute. If layer is not specified, all layers are substituted")
    parser.add_argument("--epoch_d", type = int, help="Epoch checkpoint to use.")
    parser.add_argument("--untrained_d", action = "store_true")
    parser.add_argument("--substitute_type_d", type = str, help="Type of approach to use for substitution", choices=["ALRR", "mha_only", "mha_separate_heads", "none"], default="none")
    parser.add_argument("--substitute_model_path_d", type=str, help="path to the substitue of attention. The folder should contain 6 subfolders one for each layer. Inside the FF checkpoints are stored with name: ff_network_{epoch}_layer_{layer}.pth", default = None)
    
    # Decoding related args
    args = parser.parse_args()
    # Wrapping training configuration into a dictionary
    evaluate_config = dict()
    for arg in vars(args):
        evaluate_config[arg] = getattr(args, arg)
    
    if evaluate_config["substitute_model_path"] == None:
        evaluate_config["substitute_model_path"] = os.path.join(CHECKPOINTS_SCRATCH, evaluate_config["substitute_type"] ,evaluate_config["substitute_class"])
    
    
    if evaluate_config["substitute_model_path_d"] == None:
        evaluate_config["substitute_model_path_d"] = os.path.join(CHECKPOINTS_SCRATCH, evaluate_config["substitute_type_d"] ,evaluate_config["substitute_class_d"])
    print(evaluate_config)

    # Train the original transformer model
    evaluate_transformer(evaluate_config)
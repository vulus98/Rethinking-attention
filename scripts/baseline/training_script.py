"""
    Notes:
        * I won't add model checkpoint averaging as mentioned in the paper - it just feels like an arbitrary heuristic
         and it won't add anything to the learning experience this repo aims to provide.

"""


import argparse
import time


import torch
from torch import nn
from torch.optim import Adam

# Handle imports from utils
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from utils.optimizers_and_distributions import CustomLRAdamOptimizer, LabelSmoothingDistribution
from models.definitions.transformer_model import Transformer
from utils.data_utils import get_data_loaders, get_masks_and_count_tokens, get_src_and_trg_batches, DatasetType, LanguageDirection
import utils.utils as utils
from utils.constants import *
from utils.full_sentence_utils import substitute_attention

# Global vars for logging purposes
num_of_trg_tokens_processed = 0
bleu_scores = []
global_train_step, global_val_step = [0, 0]


# Simple decorator function so that I don't have to pass these arguments every time I call get_train_val_loop
def get_train_val_loop(baseline_transformer, custom_lr_optimizer, kl_div_loss, label_smoothing, pad_token_id, time_start):

    def train_val_loop(is_train, token_ids_loader, epoch):
        global num_of_trg_tokens_processed, global_train_step, global_val_step

        if is_train:
            baseline_transformer.train()
        else:
            baseline_transformer.eval()

        device = next(baseline_transformer.parameters()).device

        #
        # Main loop - start of the CORE PART
        #
        for batch_idx, token_ids_batch in enumerate(token_ids_loader):
            src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt = get_src_and_trg_batches(token_ids_batch)
            src_mask, trg_mask, num_src_tokens, num_trg_tokens = get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch_input, pad_token_id, device)

            # log because the KL loss expects log probabilities (just an implementation detail)
            predicted_log_distributions = baseline_transformer(src_token_ids_batch, trg_token_ids_batch_input, src_mask, trg_mask)
            smooth_target_distributions = label_smoothing(trg_token_ids_batch_gt)  # these are regular probabilities

            if is_train:
                custom_lr_optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph

            loss = kl_div_loss(predicted_log_distributions, smooth_target_distributions)

            if is_train:
                loss.backward()  # compute the gradients for every trainable weight in the computational graph
                custom_lr_optimizer.step()  # apply the gradients to weights

            # End of CORE PART

            #
            # Logging and metrics
            #

            if is_train:
                global_train_step += 1
                num_of_trg_tokens_processed += num_trg_tokens

                if training_config['console_log_freq'] is not None and batch_idx % training_config['console_log_freq'] == 0:
                    print(f'Transformer training: time elapsed= {(time.time() - time_start):.2f} [s] '
                          f'| epoch={epoch + 1} | batch= {batch_idx + 1} '
                          f'| target tokens/batch= {num_of_trg_tokens_processed / training_config["console_log_freq"]}')

                    num_of_trg_tokens_processed = 0

                # Save model checkpoint
                if training_config['checkpoint_freq'] is not None and (epoch + 1) % training_config['checkpoint_freq'] == 0 and batch_idx == 0:
                    ckpt_model_name = f"transformer_ckpt_epoch_{epoch + 1}.pth"
                    torch.save(utils.get_training_state(training_config, custom_lr_optimizer.current_step_number, baseline_transformer), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))
            else:
                global_val_step += 1

    return train_val_loop


def train_transformer(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!
    # device = "cpu"
    # Step 1: Prepare data loaders
    # NOTE: If we wanted to load the pretrained transformer, we would need to first load the entire training data to get the full vocabulary. Then reload the dataset filtering for sentences s.t. S <= MAX_LEN
    train_token_ids_loader, val_token_ids_loader, test_token_ids_loader, src_field_processor, trg_field_processor = get_data_loaders(
        training_config['dataset_path'],
        training_config['language_direction'],
        training_config['dataset_name'],
        training_config['batch_size'],
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
    # model_path = os.path.join(BINARIES_PATH, training_config['model_name'])
    # model_state = torch.load(model_path)
    # baseline_transformer.load_state_dict(model_state["state_dict"], strict=True)
    # baseline_transformer.train()

    # reloading the data, filtering sentences of len>50
    train_token_ids_loader, val_token_ids_loader, test_token_ids_loader, src_field_processor, trg_field_processor = get_data_loaders(
        training_config['dataset_path'],
        training_config['language_direction'],
        training_config['dataset_name'],
        training_config['batch_size'],
        device,
        max_len_train=MAX_LEN)
    # Step 3: substitute attention
    if training_config["substitute_type"] != "none":
        substitute_attention(baseline_transformer, 
                             training_config["substitute_class"], 
                             training_config["substitute_model_path"], 
                             training_config["layer"],
                             training_config["epoch"],
                             training_config["substitute_type"],
                             training_config["untrained"]) 
    else:
        print("#"*100)
        print("\n\t NO SUBSTITUTION \n")
        print("#"*100)

    #baseline_transformer=torch.nn.DataParallel(baseline_transformer,device_ids=list(range(4)))
    # Step 3: Prepare other training related utilities
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')  # gives better BLEU score than "mean"

    # Makes smooth target distributions as opposed to conventional one-hot distributions
    # My feeling is that this is a really dummy and arbitrary heuristic but time will tell.
    label_smoothing = LabelSmoothingDistribution(BASELINE_MODEL_LABEL_SMOOTHING_VALUE, pad_token_id, trg_vocab_size, device)

    # Make resuming the training possible
    steps_taken = 0
    if (training_config['start_point']):
        ckpt_model_name = f"transformer_ckpt_epoch_{training_config['start_point']}.pth"
        path = os.path.join(CHECKPOINTS_PATH, ckpt_model_name)
        print(f"Trying to resume training from starting point {path}.")
        if (not os.path.exists(path)):
            print(f"Requested starting point {path} does not exist.")
            return
        else:
            training_state = torch.load(path)
            assert(training_config['dataset_name'] == training_state['dataset_name'])
            assert(training_config['language_direction'] == training_state['language_direction'])
            baseline_transformer.load_state_dict(training_state['state_dict'])
            steps_taken = training_state['steps_taken']
            print("Loaded state dict, resuming training")

    # Check out playground.py for an intuitive visualization of how the LR changes with time/training steps, easy stuff.
    custom_lr_optimizer = CustomLRAdamOptimizer(
                Adam(baseline_transformer.parameters(), betas=(0.9, 0.98), eps=1e-9),
                BASELINE_MODEL_DIMENSION,
                training_config['num_warmup_steps'],
                steps_taken
            )

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    train_val_loop = get_train_val_loop(baseline_transformer, custom_lr_optimizer, kl_div_loss, label_smoothing, pad_token_id, time.time())

    # Step 4: Start the training
    for epoch in range(training_config['start_point'], training_config['num_of_epochs']):
        # Training loop
        train_val_loop(is_train=True, token_ids_loader=train_token_ids_loader, epoch=epoch)

        # Validation loop
        with torch.no_grad():
            train_val_loop(is_train=False, token_ids_loader=val_token_ids_loader, epoch=epoch)

            bleu_score = utils.calculate_bleu_score(baseline_transformer, val_token_ids_loader, trg_field_processor)

    # Save the latest transformer in the binaries directory
    torch.save(utils.get_training_state(training_config, custom_lr_optimizer.current_step_number, baseline_transformer), os.path.join(BINARIES_PATH, 'replacement_layer_transformer_FF_shrink_mha'))


if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    num_warmup_steps = 4000
    # Modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    # According to the paper I infered that the baseline was trained for ~19 epochs on the WMT-14 dataset and I got
    # nice returns up to epoch ~20 on IWSLT as well (nice round number)
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=20)
    # You should adjust this for your particular machine (I have RTX 2080 with 8 GBs of VRAM so 1500 fits nicely!)
    parser.add_argument("--batch_size", type=int, help="target number of tokens in a src/trg batch", default=1500)

    # Data related args
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='which dataset to use for training', default=DatasetType.IWSLT.name)
    parser.add_argument("--language_direction", choices=[el.name for el in LanguageDirection], help='which direction to translate', default=LanguageDirection.E2G.name)
    parser.add_argument("--dataset_path", type=str, help='download dataset to this path', default=DATA_DIR_PATH)
    parser.add_argument("--model_name", type=str, help="transformer model name", default=r'transformer_128.pth')

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq", default=10)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq", default=1)
    parser.add_argument("--start_point", type=int, help="checkpoint model (epoch) where to resume training from", default=0)
    parser.add_argument("--substitute_class", type=str, help="class that substitutes attention e.g. FF_large", default="FFNetwork_shrink")
    parser.add_argument("--substitute_model_path", type=str, help="path to the substitue of attention. The folder should contain 6 subfolders one for each layer. Inside the FF checkpoints are stored with name: ff_network_{epoch}_layer_{layer}.pth")
    parser.add_argument("--layer", help = "If layer is not specified, all layers are substituted", default = None)
    parser.add_argument("--epoch", type = int, help="Epoch checkpoint to use.", default=20)
    parser.add_argument("--substitute_type", type = str, help="Epoch checkpoint to use.", choices=["mha_full", "mha_only", "mha_separate_heads", "none"], default="none")
    parser.add_argument("--untrained", type=bool, default = True)
    args = parser.parse_args()
    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    training_config['num_warmup_steps'] = num_warmup_steps

    # Train the original transformer model
    train_transformer(training_config)

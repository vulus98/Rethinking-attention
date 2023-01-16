import argparse
import time

import torch
from torch import nn
from torch.optim import Adam, SGD


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
from utils.simulator import *

configs = {
        "whole": {"batch_size": 512, "nr_units": [6, 4, 7], "nr_layers": 4},
        "just_attention": {"batch_size": 512, "nr_units": [7, 5, 7], "nr_layers": 4},
        "with_residual": {"batch_size": 2048, "nr_units": [5, 7, 6, 5, 5], "nr_layers": 6}
        }

def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)

def get_untrained_transformer():
    transformer = Transformer(
        model_dimension=BASELINE_MODEL_DIMENSION,
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        number_of_heads=BASELINE_MODEL_NUMBER_OF_HEADS,
        number_of_layers=BASELINE_MODEL_NUMBER_OF_LAYERS,
        dropout_probability=BASELINE_MODEL_DROPOUT_PROB
    ).to(device)
    return transformer

def get_trained_transformer():
    transformer = get_untrained_transformer()
    checkpoint = torch.load(os.path.join(BINARIES_PATH, "transformer_128.pth"))
    transformer.load_state_dict(checkpoint['state_dict'])
    return transformer

def replace_encoder(transformer, new_module):
    transformer.encoder = new_module

def replace_encoder_just_attention(transformer, layers):
    restructure_encoder_layers(transformer)
    for i in range(6):
        transformer.encoder.encoder_layers[i].sublayer_zero.layer = layers[i]

def replace_encoder_with_residual(transformer, layers):
    restructure_encoder_layers(transformer)
    for i in range(6):
        transformer.encoder.encoder_layers[i].sublayer_zero = layers[i]

def insert_untrained(transformer, module, ext_pref):
    if isinstance(module, list):
        for m in module:
            reset_all_weights(m)
    else:
        reset_all_weights(module)
    insert(transformer, module, ext_pref)

def insert(transformer, module, ext_pref):
    if ext_pref == "just_attention":
        replace_encoder_just_attention(transformer, module)
    elif ext_pref == "with_residual":
        replace_encoder_with_residual(transformer, module)
    elif ext_pref == "encoder":
        replace_encoder(transformer, module)
    else:
        raise f"Unknown ext_pref {ext_pref}"

def evaluate(transformer):
    transformer.eval()

    with torch.no_grad():
        bleu = utils.calculate_bleu_score(transformer, val_token_ids_loader, trg_field_processor)
    return bleu

def train(transformer, num_of_epochs, name, get_optimizer = lambda t: CustomLRAdamOptimizer(Adam(t.parameters(), betas=(0.9, 0.98), eps=1e-9), BASELINE_MODEL_DIMENSION, 4000)):
    optimizer = get_optimizer(transformer)
    time_start = time.time()

    kl_div_loss = nn.KLDivLoss(reduction='batchmean')  # gives better BLEU score than "mean"

    label_smoothing = LabelSmoothingDistribution(BASELINE_MODEL_LABEL_SMOOTHING_VALUE, pad_token_id, trg_vocab_size, device)

    for epoch in range(num_of_epochs):
        # Training loop
        transformer.train()
        for batch_idx, token_ids_batch in enumerate(train_token_ids_loader):
            src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt = get_src_and_trg_batches(token_ids_batch)
            src_mask, trg_mask, num_src_tokens, num_trg_tokens = get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch_input, pad_token_id, device)

            # log because the KL loss expects log probabilities (just an implementation detail)
            predicted_log_distributions = transformer(src_token_ids_batch, trg_token_ids_batch_input, src_mask, trg_mask)
            smooth_target_distributions = label_smoothing(trg_token_ids_batch_gt)  # these are regular probabilities

            optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph

            loss = kl_div_loss(predicted_log_distributions, smooth_target_distributions)

            loss.backward()  # compute the gradients for every trainable weight in the computational graph
            optimizer.step()  # apply the gradients to weights

            if batch_idx % 350 == 0:
                print(f'TRAIN: time elapsed={(time.time() - time_start):.2f} [s] '
                        f'| epoch={epoch + 1} | batch_idx={batch_idx} | training_loss: {loss.item():.4f}')
        # Validation loop
        with torch.no_grad():
            transformer.eval()
            bleu_score = utils.calculate_bleu_score(transformer, val_token_ids_loader, trg_field_processor)
        ckpt_model_name = f"evaluate_{name}_{epoch + 1}.pth"
        torch.save(transformer.state_dict(), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))

def train_from_scratch(transformer, num_of_epochs, name):
    reset_all_weights(transformer)
    train(transformer, num_of_epochs, f"{name}_from_scratch")

def fine_tune(transformer, name):
    train(transformer, 10, f"{name}_fine_tune", lambda t: SGD(t.parameters(), lr=0.01, momentum=0.9))

def treatment(t, name):
    bleu = {}
    print(f"{name}: Evaluating the pretrained version.")
    bleu["pretrained"] = evaluate(t)
    print(f"{name}: Fine-tuning the pretrained version.")
    fine_tune(t, name)
    bleu["fine_tuning"] = evaluate(t)
    print(f"{name}: Training everything from scratch.")
    train_from_scratch(t, 20, name)
    bleu["from_scratch"] = evaluate(t)
    for k,v in bleu.items():
        print(f"BLEU {k}:\t{v}")
    

def get_attention_sims(ext_pref):
    sims = []
    for i in range(6):
        nr_layers = configs[ext_pref]["nr_layers"]
        nr_units = configs[ext_pref]["nr_units"]
        batch_size = configs[ext_pref]["batch_size"]
        a = AttentionSimulator(nr_layers, nr_units).to(device)
        # load weights
        ckpt_model_name = get_checkpoint_name(a.name, batch_size, i, "norm" if i == 5 and ext_pref == "whole" else i, 25, ext_pref)
        model_state_dict, _ = torch.load(os.path.join(CHECKPOINTS_PATH, ckpt_model_name), map_location=device)
        a.load_state_dict(model_state_dict)
        if ext_pref != "whole":
            a = SimulatorAdapter(a).to(device)
        sims.append(a)
    return sims

def vanilla():
    print("VANILLA: Evaluating the pretrained version.")
    evaluate(get_trained_transformer())

def single_sim():
    # this was the best configuration found
    a = AttentionSimulator(4, [7, 7, 7]).to(device)
    # load weights
    ckpt_model_name = get_checkpoint_name(a.name, 1024, 0, "norm", 40, "whole")
    model_state_dict, _ = torch.load(os.path.join(CHECKPOINTS_PATH, ckpt_model_name), map_location=device)
    a.load_state_dict(model_state_dict)

    adapter = SimulatorAdapter(a).to(device)
    t = get_trained_transformer()
    insert(t, adapter, "encoder")
    treatment(t, "SINGLESIM")

def multi():
    sims = get_attention_sims("whole")
    ms = MultipleSimulator(sims).to(device)

    t = get_trained_transformer()
    insert(t, ms, "encoder")
    treatment(t, "MULTIPLESIMULATOR")
    # load fine-tuned weights
    ckpt_model_name = f"{ms.name}_lr0.0003_ckpt_epoch_5.pth"
    model_state_dict, _ = torch.load(os.path.join(CHECKPOINTS_PATH, ckpt_model_name), map_location=device)
    ms.load_state_dict(model_state_dict)
    t = get_trained_transformer()
    insert(t, ms, "encoder")
    # not necessary to train from scratch again
    print(f"MULTIPLESIMULATOR: Evaluating the pretrained fine-tuned (outside of transformer) version.")
    evaluate(t)

def just_attention():
    layers = get_attention_sims("just_attention")
    t = get_trained_transformer()
    insert(t, layers, "just_attention")
    treatment(t, "JUST ATTENTION")

def with_residual():
    layers = get_attention_sims("with_residual")
    t = get_trained_transformer()
    insert(t, layers, "with_residual")
    treatment(t, "WITH RESIDUAL")

def run():
    if (config["vanilla"]):
        vanilla()
    if (config["single_sim"]):
        single_sim()
    if (config["multi"]):
        multi()
    if (config["just_attention"]):
        just_attention()
    if (config["with_residual"]):
        with_residual()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="target number of tokens in a src/trg batch", default=1500)

    # Data related args
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='which dataset to use for training', default=DatasetType.IWSLT.name)
    parser.add_argument("--language_direction", choices=[el.name for el in LanguageDirection], help='which direction to translate', default=LanguageDirection.E2G.name)
    parser.add_argument("--dataset_path", type=str, help='download dataset to this path', default=DATA_DIR_PATH)
    parser.add_argument("--vanilla", action="store_true")
    parser.add_argument("--single_sim", action="store_true")
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--just_attention", action="store_true")
    parser.add_argument("--with_residual", action="store_true")

    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    global device, config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    global train_token_ids_loader, val_token_ids_loader, test_token_ids_loader, src_field_processor, trg_field_processor
    train_token_ids_loader, val_token_ids_loader, test_token_ids_loader, src_field_processor, trg_field_processor = get_data_loaders(
        config['dataset_path'],
        config['language_direction'],
        config['dataset_name'],
        config['batch_size'],
        device)
    global src_vocab_size, trg_vocab_size, pad_token_id
    pad_token_id = src_field_processor.vocab.stoi[PAD_TOKEN]  # pad token id is the same for target as well
    src_vocab_size = len(src_field_processor.vocab)
    trg_vocab_size = len(trg_field_processor.vocab)
    run()

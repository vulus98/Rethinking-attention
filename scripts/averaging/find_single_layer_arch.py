import argparse
import time
import os

import torch
from torch import nn
from torch.optim import Adam
import optuna

# Handle imports from utils
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from utils.constants import *
from utils.simulator import *

num_epochs = 25
print("20")
def train_model(trial, train_data_set, val_data_set, ext_pref, device):

    nr_layers = trial.suggest_int("nr_layers", 2, 6)
    nr_units = [trial.suggest_int(f"nr_units_{i}", 2, 7) for i in range(nr_layers-1)]
    batch_size_exp = trial.suggest_int("batch_size_exp", 8, 12)
    batch_size = 2**batch_size_exp

    index_in = train_data_set.index_in
    index_out = train_data_set.index_out

    model = AttentionSimulator(nr_layers = nr_layers, nr_units = nr_units).to(device)

    print(f"Starting to train model {model.name} with batch size {batch_size} from {ext_pref} layer {index_in} to layer {index_out}")
    time_start = time.time()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters())

    # to put the error in context
    train_out_magnitude = torch.mean(torch.abs(train_data_set.output))
    val_out_magnitude = torch.mean(torch.abs(val_data_set.output))
    print(f"train_out_magnitude: {train_out_magnitude}")
    print(f"val_out_magnitude: {val_out_magnitude}")

    val_loss = 0.0
    train_l = get_batches(train_data_set, batch_size)
    for epoch in range(num_epochs):
        epoch_start = time.time()
        # Training
        model.train()
        for (batch_idx, (fr, to)) in enumerate(train_l):
            inputs = train_data_set.input[fr:to]

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, train_data_set.output[fr:to])
            loss.backward()
            optimizer.step()
            if batch_idx % (len(train_l)//10) == 0:
                print(f'TRAIN: time elapsed={(time.time() - time_start):.2f} [s] '
                        f'| epoch={epoch + 1} | batch_part={int(((batch_idx+1)/len(train_l))*10)} | training_loss: {loss.item():.4f}')

        # Validation loop
        with torch.no_grad():
            model.eval()
            losses = []
            for (batch_idx, (fr, to)) in enumerate(get_batches(val_data_set, batch_size)):
                inputs = val_data_set.input[fr:to]
                outputs = model(inputs)
                loss = criterion(outputs, val_data_set.output[fr:to])
                losses.append(loss)
            losses = torch.stack(losses)
            val_loss = torch.mean(losses)
            print(f'VALIDATION loss: {val_loss:.4f} RELATIVE: {((val_loss/val_out_magnitude)*100):.2f}% in epoch {epoch+1}')

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        # Save model checkpoint
    ckpt_model_name = get_checkpoint_name(model.name, batch_size, index_in, index_out, num_epochs, ext_pref)
    torch.save((model.state_dict(), optimizer.state_dict()), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))
    return val_loss

def test_model(model, test_data_set, batch_size):
    print(f"Starting to test model {model.name}")
    test_out_magnitude = torch.mean(torch.abs(test_data_set.output))
    print(f"test_out_magnitude: {test_out_magnitude}")
    criterion = nn.MSELoss()
    with torch.no_grad():
        model.eval()
        losses = []
        for (batch_idx, (fr, to)) in enumerate(get_batches(test_data_set, batch_size)):
            inputs = test_data_set.input[fr:to]
            outputs = model(inputs)
            loss = criterion(outputs, test_data_set.output[fr:to])
            losses.append(loss)
        losses = torch.stack(losses)
        loss = torch.mean(losses)
        print(f'TEST loss: {loss:.4f} RELATIVE: {((loss/test_out_magnitude)*100):.2f}%')

def train(index_in, index_out, ext_pref):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_set = SingleWordsInterResultsDataset(index_in, index_out, "train", device, ext_pref)
    val_data_set = SingleWordsInterResultsDataset(index_in, index_out, "val", device, ext_pref)
    test_data_set = SingleWordsInterResultsDataset(index_in, index_out, "test", device, ext_pref)

    trainable = lambda t: train_model(t, train_data_set, val_data_set, ext_pref, device)

    study = optuna.create_study(direction="minimize")
    study.optimize(trainable, n_trials=30)

    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    nr_layers = trial.params["nr_layers"]
    batch_size = 2**trial.params["batch_size_exp"]
    nr_units = [0]*(nr_layers-1)
    for key, value in trial.params.items():
        if key.startswith("nr_units_"):
            nr_units[int(key[9:])] = value

    model = AttentionSimulator(nr_layers, nr_units).to(device)

    ckpt_model_name = get_checkpoint_name(model.name, batch_size, index_in, index_out, num_epochs, ext_pref)
    model_state, _ = torch.load(os.path.join(CHECKPOINTS_PATH, ckpt_model_name))

    model.load_state_dict(model_state)

    test_model(model, test_data_set, batch_size)
    del train_data_set.input
    del train_data_set.output
    del val_data_set.input
    del val_data_set.output
    del test_data_set.input
    del test_data_set.output
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=int, help="index of the input", required=True)
    parser.add_argument("--output", type=int, help="index of the output", required=True)
    parser.add_argument("--just_attention", action="store_true")
    parser.add_argument("--with_residual", action="store_true")
    parser.add_argument("--whole", action="store_true")
    args = parser.parse_args()
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    if (config["just_attention"]):
        train(args.input, args.output, "just_attention")
    elif (config["with_residual"]):
        train(args.input, args.output, "with_residual")
    elif (config["whole"]):
        train(args.input, args.output, "whole")
    else:
        print("Specify one of the three approaches: just_attention, with_residual, whole")
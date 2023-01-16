import argparse
import time
import numpy as np
import os

import torch
from torch import nn
from torch.optim import Adam

# Handle imports from utils
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from utils.constants import *

from utils.simulator import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 25
configs = {
        "whole": {"batch_size": 512, "nr_units": [6, 4, 7], "nr_layers": 4},
        "just_attention": {"batch_size": 512, "nr_units": [7, 5, 7], "nr_layers": 4},
        "with_residual": {"batch_size": 2048, "nr_units": [5, 7, 6, 5, 5], "nr_layers": 6}
        }

def train_model(index, t, config):
    index_in = index
    index_out = "norm" if index == 5 and t == "whole" else index

    # best configuration found
    nr_layers = config["nr_layers"]
    nr_units = config["nr_units"]
    batch_size = config["batch_size"]

    model = AttentionSimulator(nr_layers = nr_layers, nr_units = nr_units).to(device)

    ckpt_model_name = get_checkpoint_name(model.name, batch_size, index_in, index_out, num_epochs, t)
    checkpoint_path = os.path.join(CHECKPOINTS_PATH, ckpt_model_name)

    train_data_set = SingleWordsInterResultsDataset(index_in, index_out, "train", device, t)
    val_data_set = SingleWordsInterResultsDataset(index_in, index_out, "val", device, t)
    test_data_set = SingleWordsInterResultsDataset(index_in, index_out, "test", device, t)

    print(f"Starting to train model {model.name} with batch size {batch_size} from layer {index_in} to layer {index_out}")
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

    # Save model checkpoint
    torch.save((model.state_dict(), optimizer.state_dict()), checkpoint_path)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--whole", action="store_true")
    parser.add_argument("--just_attention", action="store_true")
    parser.add_argument("--with_residual", action="store_true")
    args = parser.parse_args()
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    if (config["whole"]):
        for i in range(6):
            train_model(i, "whole", configs["whole"])
    if (config["just_attention"]):
        for i in range(6):
            train_model(i, "just_attention", configs["just_attention"])
    if (config["with_residual"]):
        for i in range(6):
            train_model(i, "with_residual", configs["with_residual"])

if __name__ == "__main__":
    train()

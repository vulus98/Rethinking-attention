import argparse
import time
import numpy as np
import os

import torch
from torch import nn
from torch.optim import Adam

import utils.utils as utils
from utils.constants import *

from simulator import *

def train_model(index, device):
    index_in = index
    index_out = index

    nr_layers = 5
    nr_units = [5, 5, 3, 5]
    batch_size = 1024

    model = AttentionSimulator(nr_layers = nr_layers, nr_units = nr_units).to(device)
    inst_name = f"{model.name}_bs{batch_size}_fr{index_in}_to{index_out}"
    ckpt_model_name = f"{inst_name}_ckpt_epoch_40.pth"
    checkpoint_path = os.path.join(CHECKPOINTS_PATH, ckpt_model_name)

    if os.path.exists(checkpoint_path):
        return

    train_data_set = SingleWordsInterResultsDataset(index_in, index_out, "train", device)
    val_data_set = SingleWordsInterResultsDataset(index_in, index_out, "val", device)
    test_data_set = SingleWordsInterResultsDataset(index_in, index_out, "test", device)

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
    for epoch in range(40):
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

    del train_data_set.input
    del train_data_set.output
    del val_data_set.input
    del val_data_set.output
    del test_data_set.input
    del test_data_set.output
    torch.cuda.empty_cache()
    # Save model checkpoint
    torch.save((model.state_dict(), optimizer.state_dict()), checkpoint_path)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(6):
        train_model(i, device)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train()

import argparse
import time
import numpy as np
import os
from pickle import UnpicklingError
from functools import partial

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import utils.utils as utils
from utils.constants import *

from simulator import *
configs = {
        "whole": {"batch_size": 512, "nr_units": [6, 4, 7], "nr_layers": 4},
        "just_attention": {"batch_size": 512, "nr_units": [7, 5, 7], "nr_layers": 4},
        "with_residual": {"batch_size": 2048, "nr_units": [5, 7, 6, 5, 5], "nr_layers": 6}
        }

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sims = []
    for i in range(6):
        nr_layers = configs["whole"]["nr_layers"]
        nr_units = configs["whole"]["nr_units"]
        batch_size = configs["whole"]["batch_size"]
        a = AttentionSimulator(nr_layers, nr_units).to(device)
        # load weights
        ckpt_model_name = get_checkpoint_name(a.name, batch_size, i, "norm" if i == 5 else i, 25, "whole")
        model_state_dict, _ = torch.load(os.path.join(CHECKPOINTS_PATH, ckpt_model_name), map_location=device)
        a.load_state_dict(model_state_dict)
        sims.append(a)
    model = MultipleSimulator(sims).to(device)

    train_data_set = UnchangedDataset(0, "norm", "train", device)
    val_data_set = UnchangedDataset(0, "norm", "val", device)
    test_data_set = UnchangedDataset(0, "norm", "test", device)

    lr = 0.0003

    print(f"Starting to train model {model.name}")
    time_start = time.time()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr = lr)

    val_loss = 0.0
    for epoch in range(5):
        # Training
        model.train()
        for batch_idx, (inputs, labels, mask) in enumerate(train_data_set):
            optimizer.zero_grad()

            outputs = model(inputs, mask)

            m = mask.squeeze(dim=1).squeeze(dim=1).unsqueeze(dim=2)

            loss = criterion(outputs*m, labels*m)
            loss.backward()
            optimizer.step()
            if batch_idx % (len(train_data_set)//10) == 0:
                print(f'TRAIN: time elapsed={(time.time() - time_start):.2f} [s] '
                        f'| epoch={epoch + 1} | batch_part={int(((batch_idx+1)/len(train_data_set))*10)} | training_loss: {loss.item():.4f}')

        # Validation loop
        with torch.no_grad():
            model.eval()
            losses = []
            for batch_idx, (inputs, labels, mask) in enumerate(val_data_set):
                outputs = model(inputs, mask)

                m = mask.squeeze(dim=1).squeeze(dim=1).unsqueeze(dim=2)

                loss = criterion(outputs*m, labels*m)
                losses.append(loss)
            losses = torch.stack(losses)
            val_loss = torch.mean(losses)
            print(f'VALIDATION loss: {val_loss:.4f} in epoch {epoch+1}')

        # Save model checkpoint
        ckpt_model_name = f"{model.name}_lr{lr}_ckpt_epoch_{epoch+1}.pth"
        torch.save((model.state_dict(), optimizer.state_dict()), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))

if __name__ == "__main__":
    train()

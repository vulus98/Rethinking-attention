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

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultipleSimulator(device).to(device)

    train_data_set = UnchangedDataset(0, 5, "train", device)
    val_data_set = UnchangedDataset(0, 5, "val", device)
    test_data_set = UnchangedDataset(0, 5, "test", device)

    lr = 0.0005
    inst_name = f"{model.name}_lr{lr}"

    print(f"Starting to train model {model.name}")
    time_start = time.time()
    criterion = nn.MSELoss()
    # TODO: what learning rate/schedule
    optimizer = Adam(model.parameters(), lr = lr)

    val_loss = 0.0
    for epoch in range(40):
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
        ckpt_model_name = f"{inst_name}_ckpt_epoch_{epoch + 1}.pth"
        torch.save((model.state_dict(), optimizer.state_dict()), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))

if __name__ == "__main__":
    train()

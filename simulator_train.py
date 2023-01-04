import argparse
import time
import numpy as np
import os
from pickle import UnpicklingError

import torch
from torch import nn
from torch.optim import Adam
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import utils.utils as utils
from utils.constants import *

class FixedWordsInterResultsDataset(torch.utils.data.Dataset):
    def __init__(self, input_path, output_path, mask_path, n, t = "max"):
        print(f"Starting to load datasets from {input_path} and {output_path} and {mask_path}")
        start = time.time()

        self.n = n
        if t != "max" and t != "exact":
            raise ValueError("ERROR: t has to be either 'max' or 'exact'.")
        self.t = t
        self.input = []
        self.output = []
        if t == "max":
            self.mask = []
            mask_cache = f"{mask_path}_fixed_{n}_{t}.cache"

        in_cache = f"{input_path}_fixed_{n}_{t}.cache"
        out_cache = f"{output_path}_fixed_{n}_{t}.cache"

        if os.path.exists(in_cache) and os.path.exists(out_cache) and (t == "exact" or os.path.exists(mask_cache)):
            self.input = torch.load(in_cache)
            self.output = torch.load(out_cache)
            if t == "max":
                self.mask = torch.load(mask_cache)
                print(f"Finished loading mask dataset from cache {mask_cache}")
            print(f"Finished loading datasets from cache {in_cache} and {out_cache}")
            print(f"Loaded {len(self.output)} samples in {time.time() - start}s")
            return

        inf = open(input_path, "rb")
        outf = open(output_path, "rb")
        maskf = open(mask_path, "rb")
        try:
            while(True):
                # i represents one batch of sentences -> dim: batch size x padded sentence length x embedding size
                i = torch.from_numpy(np.load(inf))
                m = torch.from_numpy(np.load(maskf))
                m = torch.squeeze(m, dim=1)
                m = torch.squeeze(m, dim=1)
                o = torch.from_numpy(np.load(outf))
                l = torch.sum(m, dim = 1)
                for j in range(i.shape[0]):
                    if t == "max":
                        if l[j] <= n:
                            self.input.append(i[j, :n])
                            self.output.append(o[j, :n])
                            self.mask.append(m[j, :n])
                    else:
                        if l[j] == n:
                            self.input.append(i[j, :n])
                            self.output.append(o[j, :n])
        except (UnpicklingError, ValueError):
            print(f"Finished loading datasets from {input_path} and {output_path}")
            print(f"Loaded {len(self.output)} samples in {time.time() - start}s")
        finally:
            inf.close()
            outf.close()
            maskf.close()
        self.input = torch.cat(self.input, dim=0)
        self.output = torch.cat(self.output, dim=0)
        torch.save(self.input, in_cache)
        torch.save(self.output, out_cache)
        if t == "max":
            self.mask = torch.cat(self.mask, dim=0)
            torch.save(self.mask, mask_cache)

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        # if we have exactly the same length, there is no need for padding/masking
        if self.t == "exact":
            return (self.input[idx], self.output[idx])
        return (self.input[idx], self.output[idx], self.mask[idx])

    def emb_size(self):
        return self.input.shape[1]

class SingleWordsInterResultsDataset(torch.utils.data.Dataset):
    def __init__(self, input_path, output_path, mask_path):
        print(f"Starting to load datasets from {input_path} and {output_path} and {mask_path}")
        start = time.time()

        self.input = []
        self.output = []

        in_cache = f"{input_path}_single.cache"
        out_cache = f"{output_path}_single.cache"

        if os.path.exists(in_cache) and os.path.exists(out_cache):
            self.input = torch.load(in_cache)
            self.output = torch.load(out_cache)
            print(f"Finished loading datasets from cache {in_cache} and {out_cache}")
            print(f"Loaded {len(self.output)} samples (flattened) in {time.time() - start}s")
            return

        inf = open(input_path, "rb")
        outf = open(output_path, "rb")
        maskf = open(mask_path, "rb")
        try:
            while(True):
                # i represents one batch of sentences -> dim: batch size x padded sentence length x embedding size
                i = torch.from_numpy(np.load(inf))
                m = torch.from_numpy(np.load(maskf))
                # squeeze two times because the batch dimension is apparently also 1 at least once
                m = torch.squeeze(m, dim=1)
                m = torch.squeeze(m, dim=1)
                o = torch.from_numpy(np.load(outf))
                assert(not torch.any(torch.isnan(i)))
                assert(not torch.any(torch.isnan(o)))
                assert(not torch.any(torch.isnan(m)))
                l = torch.sum(m, dim = 1)
                for j, s in enumerate(i):
                    # get sentence length from mask
                    s_sum = torch.sum(s[:l[j]], dim=0)
                    for k, w in enumerate(s[:l[j]]):
                        # average of the rest of the sentence
                        # to catch division by zero
                        if (l[j]-1 == 0):
                            avg = torch.zeros_like(s_sum)
                        else:
                            avg = (s_sum-w)/(l[j]-1)
                        assert(not torch.any(torch.isnan(avg)))
                        e = torch.cat([w, avg], dim=0)
                        self.input.append(e)
                        self.output.append(o[j, k])
        except (UnpicklingError, ValueError):
            print(f"Finished loading datasets from {input_path} and {output_path}")
            print(f"Loaded {len(self.output)} samples (flattened) in {time.time() - start}s")
        finally:
            inf.close()
            outf.close()
            maskf.close()
        inf.close()
        outf.close()
        maskf.close()
        self.input = torch.stack(self.input, dim=1)
        self.output = torch.stack(self.output, dim=1)
        self.input = torch.transpose(self.input, 0, 1)
        self.output = torch.transpose(self.output, 0, 1)
        torch.save(self.input, in_cache)
        torch.save(self.output, out_cache)

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return (self.input[idx], self.output[idx])

    def emb_size(self):
        return self.input.shape[1]

class AttentionSimulator(nn.Module):
    def __init__(self, model_dimension, nr_layers, nr_units):
        super(AttentionSimulator, self).__init__()
        layers = []
        def append_layer(in_dim, out_dim):
            layers.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()))

        assert(nr_layers >= 1)
        if (nr_layers == 1):
            append_layer(2*model_dimension, model_dimension)
        elif isinstance(nr_units, int):
            append_layer(2*model_dimension, nr_units)
            for i in range(1, nr_layers-1):
                append_layer(nr_units, nr_units)
            append_layer(nr_units, model_dimension)
        else:
            assert(len(nr_units)+1 == nr_layers)
            append_layers(2*model_dimension, nr_units[0])
            for i in range(1, nr_layers-1):
                append_layers(nr_units[i-1], nr_units[i])
            append_layers(nr_units[-1], model_dimension)
        self.sequential = nn.Sequential(*layers)
        self.name = f"{nr_layers}_{nr_units}".replace(" ", "")

    def forward(self, x):
        return self.sequential(x)

def train_model(model, train_loader, val_loader, device):
    print(f"Starting to train model {model.name}")
    time_start = time.time()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters())
    for epoch in range(training_config['num_of_epochs']):
        # Training loop
        model.train()

        for batch_idx, batch_data in enumerate(train_loader):
            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if training_config['console_log_freq'] is not None and batch_idx % training_config['console_log_freq'] == 0:
                print(f'Simulator training: time elapsed={(time.time() - time_start):.2f} [s] '
                      f'| epoch={epoch + 1} | batch={batch_idx} | training_loss: {loss.item()}')

        # Validation loop
        with torch.no_grad():
            model.eval()
            losses = []
            for batch_idx, batch_data in enumerate(val_loader):
                inputs, labels = batch_data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                losses.append(loss)
            losses = torch.stack(losses)
            print(f'Simulator validation loss epoch {epoch+1}: validation_loss: {torch.mean(losses)}')

        # Save model checkpoint
        if training_config['checkpoint_freq'] is not None and (epoch + 1) % training_config['checkpoint_freq'] == 0:
            ckpt_model_name = f"{model.name}_ckpt_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), os.path.join(SCRATCH, ckpt_model_name))

def train(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_set = SingleWordsInterResultsDataset(training_config["train_input"], training_config["train_output"], training_config["train_mask"])
    val_data_set = SingleWordsInterResultsDataset(training_config["val_input"], training_config["val_output"], training_config["val_mask"])

    train_loader = DataLoader(train_data_set, batch_size = training_config["batch_size"])
    val_loader = DataLoader(val_data_set, batch_size = training_config["batch_size"])

    model = AttentionSimulator(model_dimension = 128, nr_layers = 3, nr_units = 256).to(device)
    train_model(model, train_loader, val_loader, device)
   # for nr_layers in range(1, 5):
   #     for nr_units in [2**i for i in range(0, 12-nr_layers)]:
   #         model = AttentionSimulator(model_dimension = 128, nr_layers = 3, nr_units = [128]).to(device)
   #         train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    num_warmup_steps = 4000

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=20)
    parser.add_argument("--batch_size", type=int, help="target number of tokens in a src/trg batch", default=1500)

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq", default=50)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq", default=1)
    parser.add_argument("--train_input", type=str, help="path to the training inputs", required=True)
    parser.add_argument("--train_output", type=str, help="path to the training outputs", required=True)
    parser.add_argument("--val_input", type=str, help="path to the validation inputs", required=True)
    parser.add_argument("--val_output", type=str, help="path to the validation outputs", required=True)
    parser.add_argument("--train_mask", type=str, help="path to the train src mask", required=True)
    parser.add_argument("--val_mask", type=str, help="path to the val src mask", required=True)
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    train(training_config)

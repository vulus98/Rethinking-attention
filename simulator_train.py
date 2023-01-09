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
    def __init__(self, input_path, output_path, mask_path, device):
        print(f"Starting to load datasets from {input_path} and {output_path} and {mask_path}")
        start = time.time()

        self.input = []
        self.output = []

        in_cache = f"{input_path}_single.cache"
        out_cache = f"{output_path}_single.cache"

        if os.path.exists(in_cache) and os.path.exists(out_cache):
            self.input = torch.load(in_cache).to(device)
            self.output = torch.load(out_cache).to(device)
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
        self.input = torch.transpose(self.input, 0, 1).to(device)
        self.output = torch.transpose(self.output, 0, 1).to(device)
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
        # TODO: allow for splitting according to multihead attention (e.g. 8 heads)
        super(AttentionSimulator, self).__init__()
        layers = [nn.BatchNorm1d(2*model_dimension)]
        def append_layer(in_dim, out_dim, dropout = False):
            layers.append(nn.Sequential(nn.Linear(int(in_dim*model_dimension), int(out_dim*model_dimension)), nn.LeakyReLU()))
            if dropout:
                layers.append(nn.Dropout(p=0.8))

        assert(nr_layers >= 1)
        if (nr_layers == 1):
            append_layer(2, 1)
        elif isinstance(nr_units, int):
            append_layer(2, nr_units)
            for i in range(1, nr_layers-1):
                append_layer(nr_units, nr_units)
            append_layer(nr_units, 1)
        else:
            assert(len(nr_units)+1 == nr_layers)
            append_layer(2, nr_units[0])
            for i in range(1, nr_layers-1):
                append_layer(nr_units[i-1], nr_units[i])
            append_layer(nr_units[-1], 1)
        self.sequential = nn.Sequential(*layers)
        self.name = f"{nr_layers}_{nr_units}".replace(" ", "")

    def forward(self, x):
        return self.sequential(x)

def train_model(model, train_data_set, val_data_set, batch_size):
    print(f"Starting to train model {model.name}")
    time_start = time.time()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters())

    train_l = [(i, min(i+batch_size, len(train_data_set)-1)) for i in range(0, len(train_data_set), batch_size)]
    val_l = [(i, min(i+batch_size, len(val_data_set)-1)) for i in range(0, len(val_data_set), batch_size)]

    # to put the error in context
    train_out_magnitude = torch.mean(torch.abs(train_data_set.output))
    val_out_magnitude = torch.mean(torch.abs(val_data_set.output))
    print(f"train_out_magnitude: {train_out_magnitude}")
    print(f"val_out_magnitude: {val_out_magnitude}")

    for epoch in range(training_config['num_of_epochs']):
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
            for (batch_idx, (fr, to)) in enumerate(val_l):
                inputs = val_data_set.input[fr:to]
                outputs = model(inputs)
                loss = criterion(outputs, val_data_set.output[fr:to])
                losses.append(loss)
            losses = torch.stack(losses)
            loss = torch.mean(losses)
            print(f'VALIDATION loss: {loss:.4f} RELATIVE: {((loss/val_out_magnitude)*100):.2f}% in epoch {epoch+1}')

        # Save model checkpoint
        if training_config['checkpoint_freq'] is not None and (epoch + 1) % training_config['checkpoint_freq'] == 0:
            ckpt_model_name = f"{model.name}_ckpt_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))

def train(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_input = training_config["input"]+"_train"
    val_input = training_config["input"]+"_val"

    train_output = training_config["output"]+"_train"
    val_output = training_config["output"]+"_val"

    train_mask = training_config["mask"]+"_train"
    val_mask = training_config["mask"]+"_val"

    train_data_set = SingleWordsInterResultsDataset(train_input, train_output, train_mask, device)
    val_data_set = SingleWordsInterResultsDataset(val_input, val_output, val_mask, device)

    model = AttentionSimulator(model_dimension = 128, nr_layers = 4, nr_units = [5, 4, 3]).to(device)
    train_model(model, train_data_set, val_data_set, training_config["batch_size"])

if __name__ == "__main__":
    num_warmup_steps = 4000

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=20)
    parser.add_argument("--batch_size", type=int, help="target number of tokens in a src/trg batch", default=1500)

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq", default=1)
    parser.add_argument("--input", type=str, help="prefix to the inputs", required=True)
    parser.add_argument("--output", type=str, help="prefix to the outputs", required=True)
    parser.add_argument("--mask", type=str, help="prefix to the src masks", required=True)
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    train(training_config)

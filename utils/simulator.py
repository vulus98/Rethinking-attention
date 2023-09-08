import time
import numpy as np
import os
import copy
from pickle import UnpicklingError

import torch
from torch import nn

import utils.utils as utils
from utils.constants import *

# Dataset with a single word and the average of its sentence as input
class SingleWordsInterResultsDataset(torch.utils.data.Dataset):
    def __init__(self, index_in, index_out, t, device, ext_pref):
        assert(t in ["train", "test", "val"])
        assert(ext_pref in ["ELR", "ALR", "ALRR"])
        pref = f"128emb_20ep_IWSLT_E2G"

        mask_path = os.path.join(LAYER_OUTPUT_PATH, f"{pref}_masks_{t}")
        input_path = os.path.join(LAYER_OUTPUT_PATH, f"{pref}_{ext_pref}_layer{index_in}_inputs_{t}")
        output_path = os.path.join(LAYER_OUTPUT_PATH, f"{pref}_{ext_pref}_layer{index_out}_outputs_{t}")

        self.index_in = index_in
        self.index_out = index_out

        print(f"Starting to load datasets from {input_path} and {output_path} and {mask_path}")
        start = time.time()

        self.input = []
        self.output = []

        in_cache = f"{input_path}_single.cache"
        out_cache = f"{output_path}_single.cache"

        if os.path.exists(in_cache) and os.path.exists(out_cache):
            self.input = torch.load(in_cache, map_location=device)
            self.output = torch.load(out_cache, map_location=device)
            print(f"Finished loading datasets from cache {in_cache} and {out_cache}")
            print(f"Loaded {len(self.output)} samples (flattened) in {time.time() - start}s")
            return

        inf = open(input_path, "rb")
        outf = open(output_path, "rb")
        maskf = open(mask_path, "rb")

        try:
            while(True):
                # input dimension: B x L x 128
                i = torch.from_numpy(np.load(inf))
                # mask dimension: B x 1 x 1 x L
                m = torch.from_numpy(np.load(maskf))
                # output dimension: B x L x 128
                o = torch.from_numpy(np.load(outf))
                m = m.squeeze(dim=1)
                m = m.squeeze(dim=1)
                denom = m.sum(dim = 1)
                avg = torch.sum(i * m.unsqueeze(2), dim=1) / denom.reshape((-1, 1))
                for j, s in enumerate(i):
                    inp = torch.cat([s[:denom[j]], avg[j].expand((denom[j], 128))], dim=1)
                    self.input.append(inp)
                    out = o[j, :denom[j]]
                    self.output.append(out)
        except (UnpicklingError, ValueError):
            print(f"Finished disk access")
            print(f"Still need to change the dataset in-memory!")
        finally:
            inf.close()
            outf.close()
            maskf.close()
        self.input = torch.cat(self.input, dim=0).to(device)
        self.output = torch.cat(self.output, dim=0).to(device)
        torch.save(self.input, in_cache)
        torch.save(self.output, out_cache)
        print(f"Loaded {len(self.output)} samples (flattened) in {time.time() - start}s")

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return (self.input[idx], self.output[idx])

# Dataset for unchanged access to the data extracted from the transformer (used in sim_all_together.py)
class UnchangedDataset(torch.utils.data.Dataset):
    def __init__(self, index_in, index_out, t, device, ext_pref):
        assert(t in ["train", "test", "val"])
        assert(ext_pref in ["ELR", "ALR", "ALRR"])
        pref = "128emb_20ep_IWSLT_E2G"

        mask_path = os.path.join(LAYER_OUTPUT_PATH, f"{pref}_masks_{t}")
        input_path = os.path.join(LAYER_OUTPUT_PATH, f"{pref}_{ext_pref}_layer{index_in}_inputs_{t}")
        output_path = os.path.join(LAYER_OUTPUT_PATH, f"{pref}_{ext_pref}_layer{index_out}_outputs_{t}")

        self.index_in = index_in
        self.index_out = index_out

        print(f"Starting to load datasets from {input_path} and {output_path} and {mask_path}")
        start = time.time()

        self.input = []
        self.output = []
        self.mask = []

        inf = open(input_path, "rb")
        outf = open(output_path, "rb")
        maskf = open(mask_path, "rb")

        try:
            while(True):
                # input dimension: B x L x 128
                i = torch.from_numpy(np.load(inf)).to(device)
                # mask dimension: B x 1 x 1 x L
                m = torch.from_numpy(np.load(maskf)).to(device)
                # output dimension: B x L x 128
                o = torch.from_numpy(np.load(outf)).to(device)
                self.input.append(i)
                self.output.append(o)
                self.mask.append(m)
        except (UnpicklingError, ValueError):
            pass
        finally:
            inf.close()
            outf.close()
            maskf.close()
        print(f"Loaded {len(self.output)} samples (flattened) in {time.time() - start}s")

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return (self.input[idx], self.output[idx], self.mask[idx])

# basically just getting the average of a sentence, but as a module
class ConvertInput(nn.Module):
    def __init__(self):
        super(ConvertInput, self).__init__()

    def forward(self, src_representations_batch, src_mask):
        # input dimension: B x L x 128
        # mask dimension: B x 1 x 1 x L
        src_mask = src_mask.squeeze(dim=1)
        src_mask = src_mask.squeeze(dim=1)
        denom = src_mask.sum(dim = 1)
        avg = torch.sum(src_representations_batch * src_mask.unsqueeze(2), dim=1) / denom.reshape((-1, 1))
        return avg

# The class getting as input the concatenation of a word embedding and the average of the sentence
class AttentionSimulator(nn.Module):
    def __init__(self, nr_layers, nr_units):
        super(AttentionSimulator, self).__init__()
        model_dimension = 128
        layers = [nn.BatchNorm1d(2*model_dimension)]
        def append_layer(in_dim, out_dim):
            layers.append(nn.Sequential(nn.Linear(int(in_dim*model_dimension), int(out_dim*model_dimension)), nn.LeakyReLU()))

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

# Adapter to transform the input we get in a transformer to the one the AttentionSimulator expects, basically just reshaping and getting the average
class SimulatorAdapter(nn.Module):
    def __init__(self, attention_simulator):
        super(SimulatorAdapter, self).__init__()
        self.c = ConvertInput()
        self.a = attention_simulator

    def forward(self, src_representations_batch, src_mask):
        B = src_representations_batch.shape[0]
        L = src_representations_batch.shape[1]

        avg = self.c(src_representations_batch, src_mask)
        x = src_representations_batch.reshape((B*L, 128))
        y = avg.unsqueeze(1).expand((-1, L, -1)).reshape((B*L, 128))
        src_representations_batch = self.a(torch.cat((x, y), dim=1)).reshape((B, L, 128))
        return src_representations_batch

# Class to replace the whole encoder with the AttentionSimulators of the "whole" approach. The last layer should also be trained to the norm of the last encoder layer.
class MultipleSimulator(nn.Module):
    def __init__(self, sims):
        super(MultipleSimulator, self).__init__()
        self.layers = nn.ModuleList()
        for a in sims:
            self.layers.append(SimulatorAdapter(a))
        self.name = f"MultipleSimulator_{self.layers[0].a.name}"

    def forward(self, src_embeddings_batch, src_mask):
        src_representations_batch = src_embeddings_batch
        for layer in self.layers:
            src_representations_batch = layer(src_representations_batch, src_mask)

        return src_representations_batch

def get_batches(data_set, batch_size):
    return [(i, min(i+batch_size, len(data_set)-1)) for i in range(0, len(data_set), batch_size)]

def get_checkpoint_name(model_name, batch_size, index_in, index_out, epoch, ext_pref):
    assert(ext_pref in ["ELR", "ALR", "ALRR"])
    inst_name = f"{model_name}_bs{batch_size}_fr{index_in}_to{index_out}_{ext_pref}"
    ckpt_model_name = f"{inst_name}_ckpt_epoch_{epoch}.pth"
    return ckpt_model_name

# replace one encoder layer
class EncoderLayerSubstituteJ(nn.Module):
    def __init__(self, encoder_layer, new_layer):
        super().__init__()
        self.sublayer_zero = SublayerSubstituteJ(encoder_layer.sublayers[0], new_layer)
        self.sublayers = encoder_layer.sublayers
        self.pointwise_net = encoder_layer.pointwise_net
        self.model_dimension = encoder_layer.model_dimension
        self.mha = encoder_layer.multi_headed_attention

    def forward(self, src_representations_batch, src_mask):
        src_representations_batch = self.sublayer_zero(src_representations_batch, src_mask)
        src_representations_batch = self.sublayers[1](src_representations_batch, self.pointwise_net)

        return src_representations_batch

# replace one sublayer
class SublayerSubstituteJ(nn.Module):
    def __init__(self, old_sublayer, layer):
        super().__init__()
        self.norm = copy.deepcopy(old_sublayer.norm)
        self.dropout = copy.deepcopy(old_sublayer.dropout)
        self.layer = copy.deepcopy(layer)

    def forward(self, representations_batch, src_mask):
        # Residual connection between input and sublayer output, details: Page 7, Chapter 5.4 "Regularization",
        return representations_batch + self.dropout(self.layer(self.norm(representations_batch), src_mask))

class MHAWrap(nn.Module):
    def __init__(self, mha):
        super().__init__()
        self.mha = copy.deepcopy(mha)

    def forward(self, srb, src_mask):
        return self.mha(srb, srb, srb, src_mask)

def restructure_encoder_layers(transformer):
    new_layers = nn.ModuleList()
    for enc_l in transformer.encoder.encoder_layers:
        new_layers.append(EncoderLayerSubstituteJ(enc_l, MHAWrap(enc_l.multi_headed_attention)))
    transformer.encoder.encoder_layers = new_layers
from pickle import UnpicklingError
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence


# Local imports
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from utils.constants import MHA_SEPARATE_CHECKPOINT_FORMAT, SCRATCH, MAX_LEN, CHECKPOINTS_SCRATCH
import models.definitions.mha_FF as nets

DATA_PATH=os.path.join(SCRATCH, "mha_outputs")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

def MAPE(target, output):
    #Mean Absolute Percentage Error

    with torch.no_grad():
        relative_error = torch.abs(output - target) / torch.max(torch.abs(target), torch.ones(output.shape, device = device)*1e-32)
        return torch.mean(relative_error)

def prepare_data(data_path, head = 0, chosen_layer = 0, batch_size = 5, t = "train", dev = False):
    if t not in ["train", "test", "val"]:
        raise ValueError("ERROR: t must be train, test, or val.")
    in_path =   os.path.join(data_path, "encoder", f"128emb_20ep_IWSLT_E2G_layer{chosen_layer}_v_inputs_{t}")
    out_path =  os.path.join(data_path, "encoder", f"128emb_20ep_IWSLT_E2G_layer{chosen_layer}_outputs_{t}")
    mask_path = os.path.join(data_path, "encoder", f"128emb_20ep_IWSLT_E2G_masks_{t}")
    dataset = SeparateHeadsDataset(in_path, out_path, mask_path, head, MAX_LEN)
    print("Training head {0}".format(head))
    if dev:
        dataset, _ = dataset = random_split(dataset, [0.2, 0.8])
    return DataLoader(dataset,  collate_fn=collate_batch, batch_size= batch_size)
    
    
def training_replacement_FF(params):
    print("Training layer {0}".format(params["num_of_curr_trained_layer"]))
    FF_net = getattr(nets, params["substitute_class"])
    for head in range(8):
        model=FF_net().to(device)
        model.train(True)
        print("FF model created")
        lr_optimizer = Adam(model.parameters(),betas=(0.9, 0.98), eps=1e-9)
        print("Preparing data")
        data_loader=prepare_data(params['dataset_path'], head=head, chosen_layer = params['num_of_curr_trained_layer'], batch_size = params["batch_size"]) 
        # TODO: loop over heads, prepare data for the head, train
        mse_loss=nn.MSELoss()
        # mean_abs_percentage_error = MeanAbsolutePercentageError()
        for epoch in range(params['num_of_epochs']):
            print("Epoch: ",epoch)
            epoch_loss=0
            num_embeddings=0
            mapes = []
            start = time.time()
            for (data,label, mask) in data_loader:
                lr_optimizer.zero_grad()
                pred=model(data,mask)
                with torch.no_grad():
                    num_embeddings+=torch.sum(torch.flatten(mask)).item()
                    loss_normalizer=torch.sum(torch.flatten(mask)).item()/(mask.shape[0]*mask.shape[1])
                loss=mse_loss(label,pred)/loss_normalizer
                loss.backward()
                lr_optimizer.step()
                with torch.no_grad():
                    epoch_loss+=loss.item()*torch.sum(torch.flatten(mask)).item()
                    mapes.append(MAPE(label, pred))
            if (epoch % 20 == 0):
                ckpt_model_name = MHA_SEPARATE_CHECKPOINT_FORMAT.format(epoch+1, params['num_of_curr_trained_layer'], head)
                torch.save(model.state_dict(), os.path.join(params["checkpoints_folder"], ckpt_model_name))
            print(f"Loss per embedding element:{epoch_loss/num_embeddings}, MAPE: {MAPE(label, pred)}, time: {time.time() - start}")

class SeparateHeadsDataset(torch.utils.data.Dataset):
    # NOTE: added h to specify which head to use
    def __init__(self, input_path, output_path, mask_path, h, n, t = "max"):
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
            mask_cache = f"{mask_path}_h_{h}_fixed_{n}_{t}.cache"

        in_cache = f"{input_path}_h_{h}_fixed_{n}_{t}.cache"
        out_cache = f"{output_path}_h_{h}_fixed_{n}_{t}.cache"

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
                            self.input.append(i[j, :l[j]])
                            self.output.append(o[j,h,:l[j]])
                            self.mask.append(m[j, :l[j]])
                    else:
                        if l[j] == n:
                            self.input.append(i[j,h ,:l[j]])
                            self.output.append(o[j, :l[j]])
        except (UnpicklingError, ValueError):
            print(f"Finished loading datasets from {input_path} and {output_path}")
            print(f"Loaded {len(self.output)} samples in {time.time() - start}s")
        finally:
            inf.close()
            outf.close()
            maskf.close()
        # self.input = torch.cat(self.input, dim=0)
        # self.output = torch.cat(self.output, dim=0)
        print(self.input[0].shape)
        torch.save(self.input, in_cache)
        torch.save(self.output, out_cache)
        if t == "max":
            # self.mask = torch.cat(self.mask, dim=0)
            torch.save(self.mask, mask_cache)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        # if we have exactly the same length, there is no need for padding/masking
        if self.t == "exact":
            return (self.input[idx], self.output[idx])
        return (self.input[idx], self.output[idx], self.mask[idx])

    def emb_size(self):
        return self.input.shape[1]
    
def pad_shape(batch, masks = False):
    shape = batch.shape
    if masks:
        return shape[0],MAX_LEN-shape[1] 
    return shape[0], MAX_LEN-shape[1], shape[2]

def collate_batch(batch):   
    """Creates a batch given a list of inputs. The output is the concatenation of the outputs from a single head for each word reperesentation in the sentece. Mask has the same shape as the output because the FF_net should multiply outputs*masks after inference. Here there is no need to multiply the inputs by masks because there is no padding related to the batch. The multiplication with the mask is performed in the AttentionSubistute because there some padding might be added when batching. 

    Args:
        batch (list): list of tuples (input(S x MD), output(S x HD), batch(S))

    Returns:
        inputs : B x MAX_LEN*MD
        outputs: B x MAX_LEN*HD
        masks  : B x MAX_LEN*HD
    """
    # Pad all elements to the same length
    inputs  = pad_sequence([x[0] for x in batch], batch_first=True, padding_value=0)
    outputs = pad_sequence([x[1] for x in batch], batch_first=True, padding_value=0)
    masks   = pad_sequence([x[2] for x in batch], batch_first=True, padding_value=0) 
    # print(inputs.shape)
    # print(outputs.shape)
    # print(masks.shape)
    
    # Pad to fixed length
    inputs = torch.cat([inputs, torch.zeros(pad_shape(inputs))], dim = 1).to(device)
    outputs = torch.cat([outputs, torch.zeros(pad_shape(outputs))], dim = 1).to(device)
    masks = torch.cat([masks, torch.zeros(pad_shape(masks, masks = True), dtype=torch.bool)], dim = 1).to(device)
    
    # Reshape concatenating the embeddings for each sentence
    masks = torch.repeat_interleave(masks, outputs.shape[-1] ,dim=1)
    inputs = torch.reshape(inputs, (inputs.shape[0],inputs.shape[1]*inputs.shape[2]))
    outputs = torch.reshape(outputs, (outputs.shape[0],outputs.shape[1]*outputs.shape[2]))
    masks = masks.reshape(outputs.shape)
    return inputs, outputs, masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=41)
    parser.add_argument("--dataset_path", type=str, help='download dataset to this path', default=DATA_PATH)
    parser.add_argument("--model_dimension", type=str, help='embedding size', default=128)
    parser.add_argument("--num_of_curr_trained_layer", type=str, help='num_of_curr_trained_layer', default=5)
    parser.add_argument("--batch_size", type=str, help='batch_size', default=2000)
    parser.add_argument("--substitute_class", type = str, help="name of the FF to train defined in models/definitions/mha_only.py", required=True)
    
    args = parser.parse_args()
    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    print("Training arguments parsed")
    training_config["checkpoints_folder"] = os.path.join(CHECKPOINTS_SCRATCH,"mha_separate_heads", training_config["substitute_class"], f"layer{training_config['num_of_curr_trained_layer']}")    
    os.makedirs(training_config["checkpoints_folder"], exist_ok = True)
    print(training_config["checkpoints_folder"])
    training_replacement_FF(training_config)

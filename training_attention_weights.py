"""
    To run for developing the networks: 
    python3 training_attention_weights.py --debug False 
"""

from pickle import UnpicklingError
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import argparse
from torch.optim import Adam
from utils.constants import MAX_LEN, CHECKPOINTS_SCRATCH, ATTENTION_WEIGHTS_OUTPUT_PATH
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import time

DATA_PATH=ATTENTION_WEIGHTS_OUTPUT_PATH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

def MAPE(target, output):
    #Mean Absolute Percentage Error
    with torch.no_grad():
        relative_error = torch.abs(output - target) / torch.max(torch.abs(target), torch.ones(output.shape, device = device)*1e-32)
        return torch.mean(relative_error)
        
# TODO: Define the network architecture to train
class FFNetwork(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        self.layers=list()
        widths=[1,1,2,2,4,4,2,2,1,1]
        self.depth=len(widths)-1
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            self.layers.extend([nn.LayerNorm(self.width//widths[i]),nn.Linear(self.width//widths[i], self.width//widths[i+1])])
            if(i<self.depth-1):
                self.layers.append(nn.LeakyReLU())
    def forward(self,data,mask):
        for layer in self.layers:
            data=layer(data)
        return data*mask
    
    def init_weights(self):
        for layer in self.layers:
            if(layer==nn.Linear):
                nn.init.uniform_(layer.weight)
                layer.bias.data.fill_(0.01)
                
def prepare_data(data_path, chosen_layer = 0, batch_size = 5, t = "train"): 
    if t not in ["train", "test", "val"]:
        raise ValueError("ERROR: t must be train, test, or val.")
    in_path_q =   os.path.join(data_path,f"128emb_20ep_IWSLT_E2G_layer{chosen_layer}_q_inputs_{t}")
    in_path_k =   os.path.join(data_path,f"128emb_20ep_IWSLT_E2G_layer{chosen_layer}_k_inputs_{t}")
    
    out_path =  os.path.join(data_path,f"128emb_20ep_IWSLT_E2G_layer{chosen_layer}_outputs_{t}")
    mask_path = os.path.join(data_path,f"128emb_20ep_IWSLT_E2G_masks_{t}")
    dataset = AttentionWeightsDataset(in_path_q,in_path_k, out_path, mask_path, MAX_LEN)
    
    return DataLoader(dataset,  collate_fn=collate_batch, batch_size= batch_size)
    
# TODO modify the training loop. For each input/output we shoul iterate over 
def training_replacement_FF(params):
    model=FFNetwork().to(device)
    #model.init_weights()
    model.train(True)
    print("FF model created")
    lr_optimizer = Adam(model.parameters(), lr=0.0001,betas=(0.9, 0.98), eps=1e-9)
    print("Preparing data")
    t = "val" if params["debug"] else "train" # this is useful for debugging to train only on the small validation dataset
    data_loader=prepare_data(params['dataset_path'], chosen_layer = params['num_of_curr_trained_layer'], batch_size = params["batch_size"], t = t) 
      
    mse_loss=nn.MSELoss()
    # mean_abs_percentage_error = MeanAbsolutePercentageError()
    for epoch in range(params['num_of_epochs']):
        print("Epoch: ",epoch)
        epoch_loss=0
        num_embeddings=0
        mapes = []
        start = time.time()
        for (query,key, label, mask) in data_loader:
            print(query.shape)
            print(key.shape)
            print(mask.shape)
            # lr_optimizer.zero_grad()
            # pred=model(data,mask)
            # with torch.no_grad():
            #     num_embeddings+=torch.sum(torch.flatten(mask)).item()
            #     loss_normalizer=torch.sum(torch.flatten(mask)).item()/(mask.shape[0]*mask.shape[1])
            # loss=mse_loss(label,pred)/loss_normalizer
            # loss.backward()
            # lr_optimizer.step()
            # with torch.no_grad():
            #     epoch_loss+=loss.item()*torch.sum(torch.flatten(mask)).item()
            #     mapes.append(MAPE(label, pred))
        # if epoch % 20 == 0:
        #     ckpt_model_name = f"transformer_ckpt_epoch_{epoch + 1}.pth"
        #     torch.save(model.state_dict(), os.path.join(params["checkpoints_folder"], ckpt_model_name))
        # print(f"Loss per embedding element:{epoch_loss/num_embeddings}, MAPE: {MAPE(label, pred)}, time: {time.time() - start}")

class AttentionWeightsDataset(torch.utils.data.Dataset):
    def __init__(self, input_path_q, input_path_k, output_path, mask_path, n, t = "max"):
        print(f"Starting to load datasets from \nQueries:\t{input_path_q}\nKeys:\t{input_path_k}\nTarget:\t{output_path}\nMasks:{mask_path}")
        start = time.time()

        self.n = n
        if t != "max" and t != "exact":
            raise ValueError("ERROR: t has to be either 'max' or 'exact'.")
        self.t = t
        self.input_q = []
        self.input_k = []
        self.output = []
        self.mask = []
        in_cache_q = f"{input_path_q}.cache"
        in_cache_k = f"{input_path_k}.cache"
        out_cache = f"{output_path}.cache"
        mask_cache = f"{mask_path}.cache"

        if os.path.exists(in_cache_q) and os.path.exists(in_cache_k) and os.path.exists(out_cache) and (t == "exact" or os.path.exists(mask_cache)):
            self.input_q = torch.load(in_cache_q)
            self.input_k = torch.load(in_cache_k)
            self.output = torch.load(out_cache)
            self.masks = torch.load(mask_cache)
            
            print(f"Finished loading datasets from")
            print(f"Queries:\t{in_cache_q}\nKeys:\t{in_cache_k}\nTarget:\t{out_cache}")
            print(f"Loaded {len(self.output)} samples in {time.time() - start}s")
            return

        inf_q = open(input_path_q, "rb")
        inf_k = open(input_path_k, "rb")
        outf = open(output_path, "rb")
        maskf = open(mask_path, "rb")
        try:
            while(True):
                # i represents one batch of sentences -> dim: batch size x padded sentence length x embedding size
                i_q = torch.from_numpy(np.load(inf_q))
                i_k = torch.from_numpy(np.load(inf_k))
              
                m = torch.from_numpy(np.load(maskf))
                m = torch.squeeze(m, dim=1)
                m = torch.squeeze(m, dim=1)
                o = torch.from_numpy(np.load(outf))
                l = torch.sum(m, dim = 1)
                for j in range(i_k.shape[0]):
                    if l[j] < MAX_LEN:
                        #store values without padding related to the batch
                        self.input_q.append(i_q[j, :l[j]]) 
                        self.input_k.append(i_k[j, :l[j]]) 
                        self.output.append(o[j,:,:l[j], :l[j]]) #element j in the batch, all heads, S x S limited to the unpadded entries
                        self.mask.append(m[j,  :l[j]])
        except (UnpicklingError, ValueError): 
            print(f"Finished loading datasets from ")
            print(f"Queries:\t{input_path_q}\nKeys:\t{input_path_k}\nTarget:\t{output_path}")
            print(f"Loaded {len(self.output)} samples in {time.time() - start}s")
        finally:
            inf_q.close()
            inf_k.close()
            outf.close()
            maskf.close()
        # self.input = torch.cat(self.input, dim=0)
        # self.output = torch.cat(self.output, dim=0)
        torch.save(self.input_q, in_cache_q)
        torch.save(self.input_k, in_cache_k)
        torch.save(self.output, out_cache)
        torch.save(self.mask, mask_cache)

    def __len__(self):
        return len(self.input_q)

    def __getitem__(self, idx):
        # Mask is returned, but it should contain all True values since padding was removed.
        return (self.input_q[idx],self.input_k[idx], self.output[idx])

    def emb_size(self):
        return self.input.shape[1]
    
def pad_shape(batch, masks = False):
    shape = batch.shape
    if masks:
        return shape[0],MAX_LEN-shape[1] 
    return shape[0], MAX_LEN-shape[1], shape[2]

def collate_batch(batch):   
    # TODO: I would print here some shapes just to understand the data the gets input in and outputed. 
    """Receives as input a list of a list of tuples (query, key, output). queries and keys are padded to the maximum sentences length in the batch. 
    Masks is created to indicated which values were padded. A mask is a one dimensional vector of length max_sentence_len containing true if the corresponding word is a real one and not padding.
    
       
    Args:
        batch (list): list of tuples (query, key, output).

    Returns:
        inputs_q: batch of queries of shape B x max_sentence_len x MD.
        inputs_k: batch of keys of shape B x max_sentence_len x MD.
        outputs : batch of attention weight of shape B x NH x max_sentence_len x max_sentence_len with zeros weights corresponding to padding. NOTE init to -inf?
        masks:    batch of masks  of shape B x max_sentence_len. It identifies the padding values where False is stored. 
    """
    # print("COLLATE")
    
    # Pad all elements to the same length
    # batch is a list of tuples (query, key, output, mask)
    # query.shape = key.shape = S x MD with MD model dimension 
    # output.shape = NH x S x S
    inputs_q  = pad_sequence([x[0] for x in batch], batch_first=True, padding_value=0)
    inputs_k  = pad_sequence([x[1] for x in batch], batch_first=True, padding_value=0)
    masks = pad_sequence([torch.ones(x[0].shape[0], dtype=torch.bool) for x in batch], batch_first=True, padding_value=0 ) 
    max_len = inputs_k.shape[1]
    # pad receives a tuple (pad_left, pad_right, pad_top, pad_bottom).
    # The weights matrix is padded with zeros to the right and at the bottom as if the sentence
    # was of max length
    outputs = pad_sequence([pad(x[2],(0, max_len - x[2].shape[-1] ,0 , max_len - x[2].shape[-1]) ,'constant', 0) for x in batch], batch_first=True, padding_value=0)
    return inputs_q, inputs_k, outputs, masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=150)
    parser.add_argument("--dataset_path", type=str, help='download dataset to this path', default=DATA_PATH)
    parser.add_argument("--model_dimension", type=str, help='embedding size', default=128)
    parser.add_argument("--num_of_loaded_files", type=str, help='num_of_loaded_files', default=20)
    parser.add_argument("--num_of_curr_trained_layer", type=str, help='num_of_curr_trained_layer', default=0)
    parser.add_argument("--batch_size", type=str, help='batch_size', default=20) #TODO we can increase the batch size when training, small is good for debug
    parser.add_argument("--checkpoints_folder_name", type = str, help="folder name relative to checkpoint folder")
    parser.add_argument("--debug", type = bool, help="folder name relative to checkpoint folder", default = False)
    
    args = parser.parse_args()
    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    print("Training arguments parsed")
    training_replacement_FF(training_config)

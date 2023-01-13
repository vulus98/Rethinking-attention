from pickle import UnpicklingError
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset, random_split
from torch.cuda.amp import GradScaler
import os
import argparse
from torch.optim import Adam
from utils.constants import SCRATCH, MAX_LEN,CHECKPOINTS_SCRATCH
from torch.nn.utils.rnn import pad_sequence
import time
import models.definitions.full_FF as nets
DATA_PATH=os.path.join(SCRATCH, "layer_outputs")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!
class FFDataset(Dataset):
    def __init__(self, data, masks, labels):
        self.data = torch.tensor(data,device=device)
        self.labels = torch.tensor(labels,device=device)
        self.masks= torch.tensor(masks,device=device)
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.masks[index]
        z = self.labels[index]
        return x, y, z
    
    def __len__(self):
        return len(self.data)

                
def prepare_data(data_path, chosen_layer = 0, batch_size = 5, t = "train", dev = False):
    if t not in ["train", "test", "val"]:
        raise ValueError("ERROR: t must be train, test, or val.")
    in_path =   os.path.join(data_path,f"128emb_20ep_IWSLT_E2G_layer{chosen_layer}_inputs_{t}")
    out_path =  os.path.join(data_path,f"128emb_20ep_IWSLT_E2G_layer{chosen_layer}_outputs_{t}")
    mask_path = os.path.join(data_path,f"128emb_20ep_IWSLT_E2G_masks_{t}")
    dataset = FixedWordsInterResultsDataset(in_path, out_path, mask_path, MAX_LEN)
    if dev:
        dataset, _ = dataset = random_split(dataset, [0.2, 0.8])
    return DataLoader(dataset,  collate_fn=collate_batch, batch_size= batch_size)
    
def prepare_data_legacy(dataset_path,num_of_loaded_files=10,chosen_layer=0,batch_size=5):
    data_label_path=os.path.join(dataset_path,"l{0}".format(chosen_layer))
    mask_path=os.path.join(dataset_path,"src_mask")
    whole_data_set=None
    whole_masks_set=None
    whole_labels_set=None

    for i in range(num_of_loaded_files):
        print("Loading batch {0}".format(i))

        new_data_batch=np.load(os.path.join(data_label_path,'input-batch-{0}.npy'.format(i)))[:,:50,:]
        model_dimension=new_data_batch.shape[2]
        new_data_batch=np.reshape(new_data_batch,newshape=(new_data_batch.shape[0],new_data_batch.shape[1]*new_data_batch.shape[2]))
        
        new_masks_batch=np.load(os.path.join(mask_path,'mask-batch-{0}.npy'.format(i)))[:,:50]
        new_masks_batch=np.repeat(new_masks_batch,model_dimension,axis=1)
        
        new_labels_batch=np.load(os.path.join(data_label_path,'output-batch-{0}.npy'.format(i)))[:,:50,:]
        new_labels_batch=np.reshape(new_labels_batch,newshape=(new_labels_batch.shape[0],new_labels_batch.shape[1]*model_dimension))

        new_data_batch=new_data_batch*new_masks_batch
        new_labels_batch=new_labels_batch*new_masks_batch
        if(i==0):
            whole_data_set=new_data_batch
            whole_masks_set=new_masks_batch
            whole_labels_set=new_labels_batch
        else:
            whole_data_set=np.concatenate((whole_data_set,new_data_batch),axis=0)
            whole_masks_set=np.concatenate((whole_masks_set,new_masks_batch),axis=0)
            whole_labels_set=np.concatenate((whole_labels_set,new_labels_batch),axis=0)


    dataset=FFDataset(whole_data_set,whole_masks_set,whole_labels_set)
    data_loader = DataLoader(dataset, batch_size)
    return data_loader

def training_replacement_FF(params):
    model=nets.FFNetwork_small().to(device)
    #model.init_weights()
    model.train(True)
    print("FF model created")
    lr_optimizer = Adam(model.parameters(), lr=0.001,betas=(0.9, 0.98), eps=1e-9)
    print("Preparing data")
    data_loader=prepare_data(params['dataset_path'], chosen_layer = params['num_of_curr_trained_layer'], batch_size = params["batch_size"]) 
    mse_loss=nn.MSELoss()
    for epoch in range(params['num_of_epochs']):
        print("Epoch: ",epoch)
        epoch_loss=0
        num_embeddings=0
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
        if(epoch%60==0):
            torch.save(model.state_dict(), os.path.join(CHECKPOINTS_SCRATCH, "layer{0}".format(params["num_of_curr_trained_layer"]),"ff_network_small_{0}".format(epoch)))
        print("Loss per embedding element: ",epoch_loss/num_embeddings)

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
                            self.input.append( i[ j, : l[j] ] )
                            self.output.append(o[ j, : l[j] ] )
                            self.mask.append(  m[ j, : l[j] ] )
                    else:
                        if l[j] == n:
                            self.input.append(i[j, :l[j]])
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
    # pad batch to a fixed length
    inputs  = pad_sequence([x[0] for x in batch], batch_first=True, padding_value=0)
    outputs = pad_sequence([x[1] for x in batch], batch_first=True, padding_value=0)
    masks   = pad_sequence([x[2] for x in batch], batch_first=True, padding_value=0)
    # pad batch to MAX_LEN
    inputs = torch.cat([inputs, torch.zeros(pad_shape(inputs))], dim = 1).to(device)
    outputs = torch.cat([outputs, torch.zeros(pad_shape(outputs))], dim = 1).to(device)
    masks = torch.cat([masks, torch.zeros(pad_shape(masks, masks = True), dtype=torch.bool)], dim = 1).to(device)
    # reshape 
    masks = torch.repeat_interleave(masks, inputs.shape[-1] ,dim=1)
    inputs = torch.reshape(inputs, (inputs.shape[0],inputs.shape[1]*inputs.shape[2]))
    outputs = torch.reshape(outputs, (outputs.shape[0],outputs.shape[1]*outputs.shape[2]))
    
    return inputs, outputs, masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=61)
    parser.add_argument("--dataset_path", type=str, help='download dataset to this path', default=DATA_PATH)
    parser.add_argument("--model_dimension", type=str, help='embedding size', default=128)
    parser.add_argument("--num_of_loaded_files", type=str, help='num_of_loaded_files', default=20)
    parser.add_argument("--num_of_curr_trained_layer", type=str, help='num_of_curr_trained_layer', default=5)
    parser.add_argument("--batch_size", type=str, help='batch_size', default=2000)
    args = parser.parse_args()
    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    print("Training arguments parsed")
    print("Training layer {0}".format(training_config["num_of_curr_trained_layer"]))
    training_replacement_FF(training_config)

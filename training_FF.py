import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import os
import argparse
from torch.optim import Adam
from utils.constants import SCRATCH_PATH
DATA_PATH=os.path.join(SCRATCH_PATH, "encoder/train")
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

class FFNetwork(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=50):
        super(FFNetwork, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        self.layers=list()
        widths=[1,1,2,8,16,32,16,8,2,1,1]
        self.depth=len(widths)-1
        self.layers = nn.ModuleList([nn.Linear(self.width//widths[i//2], self.width//widths[i//2+1]) if i%2==0 else nn.ReLU() for i in range(2*self.depth-1)])
    def forward(self,data,mask):
        for layer in self.layers:
            data=layer(data)
        return data*mask
    
    def init_weights(self):
        for layer in self.layers:
            if(layer==nn.Linear):
                nn.init.uniform_(layer.weight)
                layer.bias.data.fill_(0.01)

def prepare_data(dataset_path,num_of_loaded_files=10,chosen_layer=0,batch_size=5):
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
    model=FFNetwork().to(device)
    #model.init_weights()
    model.train(True)
    print("FF model created")
    lr_optimizer = Adam(model.parameters(), lr=0.001,betas=(0.9, 0.98), eps=1e-9)
    print("Preparing data")
    data_loader=prepare_data(params['dataset_path'], params['num_of_loaded_files'], params['num_of_curr_trained_layer'],params["batch_size"])
    mse_loss=nn.MSELoss()
    for epoch in range(params['num_of_epochs']):
        print("Epoch: ",epoch)
        epoch_loss=0
        for (data,mask,label) in data_loader:
            lr_optimizer.zero_grad()
            pred=model(data,mask)
            with torch.no_grad():
                loss_normalizer=torch.sum(torch.flatten(mask)).item()/(mask.shape[0]*mask.shape[1])
            loss=mse_loss(label,pred)/loss_normalizer
            loss.backward()
            lr_optimizer.step()
            epoch_loss+=loss.item()
        print("Loss: ",epoch_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=20)
    parser.add_argument("--dataset_path", type=str, help='download dataset to this path', default=DATA_PATH)
    parser.add_argument("--model_dimension", type=str, help='embedding size', default=128)
    parser.add_argument("--num_of_loaded_files", type=str, help='num_of_loaded_files', default=20)
    parser.add_argument("--num_of_curr_trained_layer", type=str, help='num_of_curr_trained_layer', default=0)
    parser.add_argument("--batch_size", type=str, help='batch_size', default=50)
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    print("Training arguments parsed")

    training_replacement_FF(training_config)

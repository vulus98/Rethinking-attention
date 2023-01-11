from torch import nn
from utils.constants import *
import torch

class FFNetwork_medium(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_medium, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        self.layers=list()
        widths=[1,2,2,1]
        self.depth=len(widths)-1
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            self.layers.extend([nn.LayerNorm(self.width * widths[i]),nn.Linear(self.width * widths[i], self.width * widths[i+1])])
            if(i<self.depth-1):
                self.layers.append(nn.LeakyReLU())
    def forward(self,data,mask):
        for layer in self.layers:
            data=layer(data)
        return data*mask
    

class FFNetwork_large(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_large, self).__init__()
        self.devices=list(range(torch.cuda.device_count()))
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        self.layers=list()
        widths=[1,2,8,1]
        self.depth=len(widths)-1
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            self.layers.extend([nn.LayerNorm(self.width*widths[i]).to(self.devices[i+1]),nn.Linear(self.width*widths[i], self.width*widths[i+1]).to(self.devices[i+1])])
            if(i<self.depth-1):
                self.layers.append(nn.LeakyReLU().to(self.devices[i+1]))

    def forward(self,data,mask):
        for (i,layer) in enumerate(self.layers):
            if(i%3==0):
                data=data.to(self.devices[i//3+1])
            data=layer(data)
        data=data.to(self.devices[0])
        mask=mask.to(self.devices[0])
        return data*mask
    
    def init_weights(self):
        for layer in self.layers:
            if(layer==nn.Linear):
                nn.init.uniform_(layer.weight)
                layer.bias.data.fill_(0.01)
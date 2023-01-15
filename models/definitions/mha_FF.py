from torch import nn
from utils.constants import *
import torch

class FFNetwork_shrink256(nn.ModuleList):
    def __init__(self,output_dim=800, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_shrink256, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        self.output_dim=output_dim
        self.layers=list()
        widths=[self.width,self.width//32,self.output_dim]
        self.depth=len(widths)-1
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            self.layers.extend([nn.LayerNorm(widths[i]),nn.Linear(widths[i], widths[i+1])])
            if(i<self.depth-1):
                self.layers.append(nn.LeakyReLU())
    def forward(self,data,mask):
        for layer in self.layers:
            data=layer(data)
        return data*mask

class FFNetwork_shrink128(nn.ModuleList):
    def __init__(self, output_dim=800,model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_shrink128, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        self.output_dim=output_dim
        self.layers=list()
        widths=[self.width,self.width//16,self.output_dim]
        self.depth=len(widths)-1
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            self.layers.extend([nn.LayerNorm(widths[i]),nn.Linear(widths[i], widths[i+1])])
            if(i<self.depth-1):
                self.layers.append(nn.LeakyReLU())
    def forward(self,data,mask):
        for layer in self.layers:
            data=layer(data)
        return data*mask


class FFNetwork_shrink(nn.ModuleList):
    def __init__(self, output_dim=800,model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_shrink, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        self.output_dim=output_dim
        self.layers=list()
        widths=[self.width,self.width//8,self.output_dim]
        self.depth=len(widths)-1
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            self.layers.extend([nn.LayerNorm(widths[i]),nn.Linear(widths[i], widths[i+1])])
            if(i<self.depth-1):
                self.layers.append(nn.LeakyReLU())
    def forward(self,data,mask):
        for layer in self.layers:
            data=layer(data)
        return data*mask

class FFNetwork_small(nn.ModuleList):
    def __init__(self, output_dim=800,model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_small, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        self.output_dim=output_dim
        self.layers=list()
        widths=[self.width,self.width//4,self.output_dim]
        self.depth=len(widths)-1
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            self.layers.extend([nn.LayerNorm(widths[i]),nn.Linear(widths[i], widths[i+1])])
            if(i<self.depth-1):
                self.layers.append(nn.LeakyReLU())
    def forward(self,data,mask):
        for layer in self.layers:
            data=layer(data)
        return data*mask

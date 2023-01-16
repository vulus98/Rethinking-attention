from torch import nn
from utils.constants import *
import torch

class FFNetwork_XS(nn.ModuleList):
    def __init__(self,output_dim=800, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_XS, self).__init__()
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

class FFNetwork_S(nn.ModuleList):
    def __init__(self, output_dim=800,model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_S, self).__init__()
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


class FFNetwork_L(nn.ModuleList):
    def __init__(self, output_dim=800,model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_L, self).__init__()
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

class FFNetwork_XL(nn.ModuleList):
    def __init__(self, output_dim=800,model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_XL, self).__init__()
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

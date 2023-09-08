from torch import nn
from utils.constants import *
import torch

class FFNetwork_XS(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_XS, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        self.layers=list()
        widths=[1,256,1]
        self.depth=len(widths)-1
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            self.layers.extend([nn.LayerNorm(self.width // widths[i]),nn.Linear(self.width // widths[i], self.width // widths[i+1])])
            if(i<self.depth-1):
                self.layers.append(nn.LeakyReLU())
    def forward(self,data,mask):
        for layer in self.layers:
            data=layer(data)
        return data*mask

class FFNetwork_S(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_S, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        self.layers=list()
        widths=[1,128,1]
        self.depth=len(widths)-1
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            self.layers.extend([nn.LayerNorm(self.width // widths[i]),nn.Linear(self.width // widths[i], self.width // widths[i+1])])
            if(i<self.depth-1):
                self.layers.append(nn.LeakyReLU())
    def forward(self,data,mask):
        for layer in self.layers:
            data=layer(data)
        return data*mask

class FFNetwork_M(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_M, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        self.layers=list()
        widths=[1,8,1]
        self.depth=len(widths)-1
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            self.layers.extend([nn.LayerNorm(self.width // widths[i]),nn.Linear(self.width // widths[i], self.width // widths[i+1])])
            if(i<self.depth-1):
                self.layers.append(nn.LeakyReLU())
    def forward(self,data,mask):
        for layer in self.layers:
            data=layer(data)
        return data*mask

class FFNetwork_L(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_L, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        self.layers=list()
        widths=[1,2,1]
        self.depth=len(widths)-1
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            self.layers.extend([nn.LayerNorm(self.width // widths[i]),nn.Linear(self.width // widths[i], self.width // widths[i+1])])
            if(i<self.depth-1):
                self.layers.append(nn.LeakyReLU())
    def forward(self,data,mask):
        for layer in self.layers:
            data=layer(data)
        return data*mask

class FFNetwork_XL(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_XL, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        self.layers=list()
        widths=[1,2,1]
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

class FFNetwork_decoder_XS(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_decoder_XS, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        widths=[1,256,1]
        self.depth=len(widths)-1
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            self.layers.extend([nn.LayerNorm(self.width // widths[i]),nn.Linear(self.width // widths[i], self.width // widths[i+1])])
            if(i<self.depth-1):
                self.layers.append(nn.LeakyReLU())
            else:
                self.layers[-1] = nn.Linear(self.width // widths[i], model_dimension)
                
    def forward(self,data,mask:torch.Tensor):
        padding_mask = mask[:,-1]
        mask = mask.repeat_interleave(self.model_dimension, dim = -1)
        data = torch.reshape(data, (data.shape[0],data.shape[1]*data.shape[2]))
        outputs = []
        for i in range(mask.shape[1]):
            o = data * mask[:, i, :]
            for layer in self.layers:
                o = layer(o)
            outputs.append(o * padding_mask[:, i].view((-1, 1)).repeat_interleave(self.model_dimension, dim = 1))
        return torch.stack(outputs, dim = 1)

class FFNetwork_decoder_S(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_decoder_S, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        widths=[1,128,1]
        self.depth=len(widths)-1
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            self.layers.extend([nn.LayerNorm(self.width // widths[i]),nn.Linear(self.width // widths[i], self.width // widths[i+1])])
            if(i<self.depth-1):
                self.layers.append(nn.LeakyReLU())
            else:
                self.layers[-1] = nn.Linear(self.width // widths[i], model_dimension)
                
    def forward(self,data,mask:torch.Tensor):
        padding_mask = mask[:,-1]
        mask = mask.repeat_interleave(self.model_dimension, dim = -1)
        data = torch.reshape(data, (data.shape[0],data.shape[1]*data.shape[2]))
        outputs = []
        for i in range(mask.shape[1]):
            o = data * mask[:, i, :]
            for layer in self.layers:
                o = layer(o)
            outputs.append(o * padding_mask[:, i].view((-1, 1)).repeat_interleave(self.model_dimension, dim = 1))
        return torch.stack(outputs, dim = 1)

class FFNetwork_decoder_M(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_decoder_M, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        widths=[1,8,1]
        self.depth=len(widths)-1
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            self.layers.extend([nn.LayerNorm(self.width // widths[i]),nn.Linear(self.width // widths[i], self.width // widths[i+1])])
            if(i<self.depth-1):
                self.layers.append(nn.LeakyReLU())
            else:
                self.layers[-1] = nn.Linear(self.width // widths[i], model_dimension)
                
    def forward(self,data,mask:torch.Tensor):
        padding_mask = mask[:,-1]
        mask = mask.repeat_interleave(self.model_dimension, dim = -1)
        data = torch.reshape(data, (data.shape[0],data.shape[1]*data.shape[2]))
        outputs = []
        for i in range(mask.shape[1]):
            o = data * mask[:, i, :]
            for layer in self.layers:
                o = layer(o)
            outputs.append(o * padding_mask[:, i].view((-1, 1)).repeat_interleave(self.model_dimension, dim = 1))
        return torch.stack(outputs, dim = 1)
    
class FFNetwork_decoder_L(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_decoder_L, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        widths=[1,2,1]
        self.depth=len(widths)-1
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            self.layers.extend([nn.LayerNorm(self.width // widths[i]),nn.Linear(self.width // widths[i], self.width // widths[i+1])])
            if(i<self.depth-1):
                self.layers.append(nn.LeakyReLU())
            else:
                self.layers[-1] = nn.Linear(self.width // widths[i], model_dimension)
                
    def forward(self,data,mask:torch.Tensor):
        padding_mask = mask[:,-1]
        mask = mask.repeat_interleave(self.model_dimension, dim = -1)
        data = torch.reshape(data, (data.shape[0],data.shape[1]*data.shape[2]))
        outputs = []
        for i in range(mask.shape[1]):
            o = data * mask[:, i, :]
            for layer in self.layers:
                o = layer(o)
            outputs.append(o * padding_mask[:, i].view((-1, 1)).repeat_interleave(self.model_dimension, dim = 1))
        return torch.stack(outputs, dim = 1)
    
class FFNetwork_decoder_XL(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_decoder_XL, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        widths=[1,2,1]
        self.depth=len(widths)-1
        self.layers=nn.ModuleList()
        for i in range(self.depth):
            self.layers.extend([nn.LayerNorm(self.width * widths[i]),nn.Linear(self.width * widths[i], self.width * widths[i+1])])
            if(i<self.depth-1):
                self.layers.append(nn.LeakyReLU())
            else:
                self.layers[-1] = nn.Linear(self.width * widths[i], model_dimension)
                
    def forward(self,data,mask:torch.Tensor):
        padding_mask = mask[:,-1]
        mask = mask.repeat_interleave(self.model_dimension, dim = -1)
        data = torch.reshape(data, (data.shape[0],data.shape[1]*data.shape[2]))
        outputs = []
        for i in range(mask.shape[1]):
            o = data * mask[:, i, :]
            for layer in self.layers:
                o = layer(o)
            outputs.append(o * padding_mask[:, i].view((-1, 1)).repeat_interleave(self.model_dimension, dim = 1))
        return torch.stack(outputs, dim = 1)
    
from torch import nn
from utils.constants import *
import torch

class FFNetwork_shrink256(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_shrink256, self).__init__()
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

class FFNetwork_shrink128(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_shrink128, self).__init__()
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

class FFNetwork_shrink8(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_shrink8, self).__init__()
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

class FFNetwork_shrink(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_shrink, self).__init__()
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

class FFNetwork_small(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_small, self).__init__()
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
                
class FFNetwork(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork, self).__init__()
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
        # self.ln1=nn.LayerNorm(self.width).to(devices[1])
        # self.ff1=nn.Linear(self.width, 2*self.width).to(devices[1])
        # self.nl1=nn.LeakyReLU().to(devices[1])

        # self.ln2=nn.LayerNorm(2*self.width).to(devices[2])
        # self.ff2=nn.Linear(2*self.width, 4*self.width).to(devices[2])
        # self.nl2=nn.LeakyReLU().to(devices[2])

        # self.ln3=nn.LayerNorm(4*self.width).to(devices[3])
        # self.ff3=nn.Linear(4*self.width, 8*self.width).to(devices[3])
        # self.nl3=nn.LeakyReLU().to(devices[3])

        # self.ln4=nn.LayerNorm(8*self.width).to(devices[4])
        # self.ff4=nn.Linear(8*self.width, 4*self.width).to(devices[4])
        # self.nl4=nn.LeakyReLU().to(devices[4])

        # self.ln4=nn.LayerNorm(4*self.width).to(devices[5])
        # self.ff4=nn.Linear(4*self.width, self.width).to(devices[5])
        

    def forward(self,data,mask):
        # data=data.to(devices[1])
        # data=self.ln1(data)
        # data=self.ff1(data)
        # data=self.nl1(data)

        # data=data.to(devices[2])
        # data=self.ln2(data)
        # data=self.ff2(data)
        # data=self.nl2(data)

        # data=data.to(devices[3])
        # data=self.ln3(data)
        # data=self.ff3(data)
        # data=self.nl3(data)

        # data=data.to(devices[4])
        # data=self.ln4(data)
        # data=self.ff4(data)
        # data=self.nl4(data)

        # data=data.to(devices[5])
        # data=self.ln5(data)
        # data=self.ff5(data)
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

class FFNetwork_decoder_shrink128(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_decoder_shrink128, self).__init__()
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

class FFNetwork_decoder_shrink256(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_decoder_shrink256, self).__init__()
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

class FFNetwork_decoder_shrink128(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_decoder_shrink128, self).__init__()
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

class FFNetwork_decoder_shrink8(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_decoder_shrink8, self).__init__()
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
    
class FFNetwork_decoder_shrink(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_decoder_shrink, self).__init__()
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
    
class FFNetwork_decoder_small(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_decoder_small, self).__init__()
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
    
class FFNetwork_decoder_mdeium(nn.ModuleList):
    def __init__(self, model_dimension=128,sentence_length=MAX_LEN):
        super(FFNetwork_decoder_mdeium, self).__init__()
        self.sentence_length=sentence_length
        self.model_dimension=model_dimension
        self.width=self.sentence_length*self.model_dimension
        widths=[1,2,2,1]
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
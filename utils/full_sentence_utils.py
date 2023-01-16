import math

import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from models.definitions.transformer_model import MultiHeadedAttention, Transformer

from utils.constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # checking whether you have a GPU, I hope so!

class EncoderLayerSubstitute(nn.Module):
    """This class replaces the entire sublayer logic. It gets from the second layer from the original Layer and substitutes the first one 
        with the provided FF. The first layer, in the original transformer corresponds to mha, residual connectio and layer norm.
    """
    def __init__(self, encoder_layer, FF_net, device):
        super().__init__()
        self.sublayers = encoder_layer.sublayers
        self.multi_headed_attention = encoder_layer.multi_headed_attention
        self.pointwise_net = encoder_layer.pointwise_net
        self.sublayer_zero = SublayerZeroSubstitute(FF_net, device)
        self.model_dimension = encoder_layer.model_dimension

    def forward(self, src_representations_batch, src_mask):
        # Define anonymous (lambda) function which only takes src_representations_batch (srb) as input,
        # this way we have a uniform interface for the sublayer logic.

        # Self-attention MHA sublayer followed by point-wise feed forward net sublayer
        # The first sublayer is substituted with out FF network
        # Original code in EncoderLayer:
        #   encoder_self_attention = lambda srb: self.multi_headed_attention(query=srb, key=srb, value=srb, mask=src_mask)
        #   src_representations_batch = self.sublayers[0](src_representations_batch, encoder_self_attention)
        
        src_representations_batch = self.sublayer_zero(src_representations_batch, src_mask)
        src_representations_batch = self.sublayers[1](src_representations_batch, self.pointwise_net)

        return src_representations_batch

class SublayerZeroSubstitute(torch.nn.Module):
    """Layer that adapts FF network to replace MultiHeadedAttention layers.

    Args:
        torch (nn.Module): Feed forward network that gets the concatenated values of words representation and mimics the behavior of MultiHeadedAttention
    """

    def __init__(self, FF_net, device):
        super().__init__()
        self.FFNetwork = FF_net
        self.device = device

    def forward(self, src_representations_batch, mask): 
        """
            query, key and value are all equals. The signature matches the signature of the forward method of MultiHeadedAttention. 
            query.shape = B x S x MD where B is batch size, S is sentence length and MD is model dimension
            mask.shape = B x 1 x 1 x S
        """
        mask = torch.squeeze(torch.squeeze(mask, dim = 1), dim = 1)
        output_shape = src_representations_batch.shape
        # Pad and Reshape
        src_representations_batch = torch.cat([ src_representations_batch, torch.zeros(pad_shape(src_representations_batch), device = self.device) ], dim = 1)
        intermediate_shape = src_representations_batch.shape
        mask = torch.cat([mask, torch.zeros(pad_shape(mask, masks = True), device = self.device, dtype=torch.bool) ], dim = 1)
        mask = torch.repeat_interleave(mask, src_representations_batch.shape[-1] ,dim=1)
        src_representations_batch = torch.reshape(src_representations_batch, 
                                                (src_representations_batch.shape[0],src_representations_batch.shape[1]*src_representations_batch.shape[2]))
        
        # Feed through the network
        src_representations_batch = src_representations_batch * mask
        src_representations_batch = self.FFNetwork(src_representations_batch, mask)
        
        # Reshape and unpdad
        src_representations_batch = torch.reshape(src_representations_batch, intermediate_shape)
        src_representations_batch, _ = torch.split(src_representations_batch, [output_shape[1], intermediate_shape[1] - output_shape[1]], dim = 1)
        assert src_representations_batch.shape == output_shape
        return src_representations_batch

def replace_sublayer(transformer: nn.Module, substitute: nn.Module, layer:int, device = "cuda"):
    transformer.encoder.encoder_layers[layer] = EncoderLayerSubstitute(transformer.encoder.encoder_layers[layer], substitute, device)
    return transformer

class Attention(nn.Module):
    def __init__(self, mha: MultiHeadedAttention):
        super().__init__()

        self.head_dimension = mha.head_dimension
        self.number_of_heads = mha.number_of_heads

        self.qkv_nets = mha.qkv_nets  # identity activation hence "nets"
        self.attention_dropout =  mha.attention_dropout  # no pun intended, not explicitly mentioned in paper
        self.softmax = mha.softmax  # -1 stands for apply the softmax along the last dimension

        self.log_attention_weights = mha.log_attention_weights  # should we log attention weights
        self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        # Step 1: Input linear projection
        # Notation: B - batch size, NH - number of heads, S/T - max src/trg token-sequence length, HD - head dimension
        # Shape goes from (B, S/T, NH*HD) over (B, S/T, NH, HD) to (B, NH, S/T, HD) (NH*HD=D where D is model dimension)
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]

        # Step 1: Scaled dot-product attention, Page 4, Chapter 3.2.1 "Scaled Dot-Product Attention"
        # Notation: B - batch size, S/T max src/trg token-sequence length, NH - number of heads, HD - head dimension
        # query/key/value shape = (B, NH, S/T, HD), scores shape = (B, NH, S, S), (B, NH, T, T) or (B, NH, T, S)
        # scores have different shapes as MHA is used in 3 contexts, self attention for src/trg and source attending MHA
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)

        # Step 2: Optionally mask tokens whose representations we want to ignore by setting a big negative number
        # to locations corresponding to those tokens (force softmax to output 0 probability on those locations).
        # mask shape = (B, 1, 1, S) or (B, 1, T, T) will get broad-casted (copied) as needed to match scores shape
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))

        # Step 3: Calculate the attention weights - how much should we attend to surrounding token representations
        attention_weights = self.softmax(scores)
        if self.log_attention_weights:
            self.attention_weights = attention_weights
        # Step 4: Not defined in the original paper apply dropout to attention weights as well
        attention_weights = self.attention_dropout(attention_weights)

        # Step 5: based on attention weights calculate new token representations
        # attention_weights shape = (B, NH, S, S)/(B, NH, T, T) or (B, NH, T, S), value shape = (B, NH, S/T, HD)
        # Final shape (B, NH, S, HD) for source MHAs or (B, NH, T, HD) target MHAs (again MHAs are used in 3 contexts)
        intermediate_token_representations = torch.matmul(attention_weights, value)

        return intermediate_token_representations 

class AttentionSubstituteSeparateHeads(nn.Module):
    def __init__(self, ff_list:list, device = "cuda"):
        """Substitutes each attention head with a FF.

        Args:
            ff_list (list[FF_network]): Feed forward nets that compute the attention values
        """
        super().__init__()
        self.ff_list = ff_list
        self.device = device
        
    def forward(self, query, key, value, mask):
        """This layer substitutes the Attention layer in mha2. 

        Args:
            query (Tensor): B x S x MD
            key (Tensor):  B x S x MD
            value (Tensor):  B x S x MD
            mask (Tensor): B x 1 x 1 x S

        Returns:
            Tensor: B x NH x S x HD
        """
        S = value.shape[1]
        B = len(value)
        HD = BASELINE_MODEL_DIMENSION // BASELINE_MODEL_NUMBER_OF_HEADS
        # 1. Pad to MAX_LEN
        inputs = torch.cat([value, torch.zeros(pad_shape(value), device = self.device)], dim = 1)
        inputs_shape = inputs.shape
        mask = torch.squeeze(torch.squeeze(mask, dim = 1), dim = 1)
        mask = torch.cat([mask, torch.zeros(pad_shape(mask, masks = True), device = self.device, dtype=torch.bool) ], dim = 1)
        # 2. Flatten
        mask = torch.repeat_interleave(mask, inputs.shape[-1] ,dim=1)
        inputs = inputs.reshape((inputs_shape[0], inputs_shape[1]* inputs_shape[2]))
        # 3. Compute
        inputs = inputs*mask
        mask = mask.reshape((B,MAX_LEN  * HD, BASELINE_MODEL_NUMBER_OF_HEADS)).transpose(1,2)
        outputs = []
        for h, ff in enumerate(self.ff_list):
            # inputs.shape = B x MAX_LEN * MD
            # mask.shape   = B x MAX_LEN * HD
            # outputs[i] has shape BxMAX_LENxHD, HD = head dimension
            outputs.append(ff.forward(inputs, mask[:,h]))
        # shape = BxNHxMAX_LEN*HD
        outputs = torch.stack(outputs, dim = 1)
        # 4. Unflatten and unpad
        # shape = BxNHxSxHD
        outputs = outputs.reshape((outputs.shape[0], outputs.shape[1], MAX_LEN, -1))
        outputs, padding =  torch.split(outputs,[S, MAX_LEN - S] , dim = 2)   
        # assert(np.prod(pad.shape) == (pad == 0).sum())
        return outputs 

def replace_mha_separate_heads(transformer: nn.Module, substitute: list, layer:int, device = "cuda"):
    if not type(transformer.encoder.encoder_layers[layer].multi_headed_attention) == MultiHeadedAttention2:
        raise TypeError("Use function mha_to_mha2 first")
    transformer.encoder.encoder_layers[layer].multi_headed_attention.attention = AttentionSubstituteSeparateHeads(substitute, device = device)
    return transformer

class MultiHeadedAttention2(nn.Module):
    def __init__(self, mha:MultiHeadedAttention):
            super().__init__()

            self.head_dimension = mha.head_dimension
            self.number_of_heads = mha.number_of_heads

            # Split mha into attention and linear layer in front. 
            self.attention = Attention( mha )
            self.out_projection_net = mha.out_projection_net
            
    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        
        intermediate_token_representations = self.attention(query, key, value, mask)
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1, self.number_of_heads * self.head_dimension)

        # Step 4: Output linear projection
        token_representations = self.out_projection_net(reshaped)

        return token_representations

def mha_to_mha2(transformer: Transformer, layers:list = [0,1,2,3,4,5], attention_type = "encoder"):
    """Substitutes MultiHeadedAttention with MultiHeadedAttention2. Useful to extract values before and after attention without the linear layer in the front.

    Args:
        transformer (Transformer): _description_
        layers (list): list of layers to modify. By default all mha in the encoders are substituted
    """
    if attention_type == "encoder":
        for l in layers:
            transformer.encoder.encoder_layers[l].multi_headed_attention =  MultiHeadedAttention2(transformer.encoder.encoder_layers[l].multi_headed_attention)
            
    elif attention_type == "decoder_self": 
        for l in layers:
            transformer.decoder.decoder_layers[l].trg_multi_headed_attention =  MultiHeadedAttention2(transformer.decoder.decoder_layers[l].trg_multi_headed_attention)
            
    elif attention_type == "decoder_cross":
        for l in layers:
            transformer.decoder.decoder_layers[l].src_multi_headed_attention =  MultiHeadedAttention2(transformer.decoder.decoder_layers[l].src_multi_headed_attention)
    else:
        raise ValueError("attention_type must be in ['encoder', 'decoder_self', 'decoder_cross']")

class AttentionWeights(nn.Module):
    def __init__(self, mha: MultiHeadedAttention):
        super().__init__()

        self.head_dimension = mha.head_dimension
        self.number_of_heads = mha.number_of_heads

        self.qkv_nets = mha.qkv_nets  # identity activation hence "nets"
        self.attention_dropout =  mha.attention_dropout  # no pun intended, not explicitly mentioned in paper
        self.softmax = mha.softmax  # -1 stands for apply the softmax along the last dimension

        self.log_attention_weights = mha.log_attention_weights  # should we log attention weights
        self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

    def forward(self, query, key, mask):
        batch_size = query.shape[0]

        # Step 1: Input linear projection
        # Notation: B - batch size, NH - number of heads, S/T - max src/trg token-sequence length, HD - head dimension
        # Shape goes from (B, S/T, NH*HD) over (B, S/T, NH, HD) to (B, NH, S/T, HD) (NH*HD=D where D is model dimension)
        #query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
        #                     for net, x in zip(self.qkv_nets, (query, key, value))]
        query, key = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                            for net, x in zip(self.qkv_nets[:2], (query, key))]

        # Step 1: Scaled dot-product attention, Page 4, Chapter 3.2.1 "Scaled Dot-Product Attention"
        # Notation: B - batch size, S/T max src/trg token-sequence length, NH - number of heads, HD - head dimension
        # query/key/value shape = (B, NH, S/T, HD), scores shape = (B, NH, S, S), (B, NH, T, T) or (B, NH, T, S)
        # scores have different shapes as MHA is used in 3 contexts, self attention for src/trg and source attending MHA
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)

        # Step 2: Optionally mask tokens whose representations we want to ignore by setting a big negative number
        # to locations corresponding to those tokens (force softmax to output 0 probability on those locations).
        # mask shape = (B, 1, 1, S) or (B, 1, T, T) will get broad-casted (copied) as needed to match scores shape
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))

        # Step 3: Calculate the attention weights - how much should we attend to surrounding token representations
        attention_weights = self.softmax(scores)
        if self.log_attention_weights:
            self.attention_weights = attention_weights
        # Step 4: Not defined in the original paper apply dropout to attention weights as well
        attention_weights = self.attention_dropout(attention_weights)

   
        return attention_weights 

class AttentionSubstitute(nn.Module):
    def __init__(self, FF_net:nn.Module, device = "cuda"):
        """Substitutes mha with a single FF. 

        Args:
            ff_list (): Feed forward nets that compute the attention values
        """
        super().__init__()
        self.ff = FF_net
        self.device = device
        
    def forward(self, query, key, value, mask):
        """This layer substitutes the Attention layer in mha2. 

        Args:
            query (Tensor): B x S x MD
            key (Tensor):  B x S x MD
            value (Tensor):  B x S x MD
            mask (Tensor): B x 1 x 1 x S

        Returns:
            Tensor: B x NH x S x HD
        """
        S = value.shape[1]
        B = len(value)
        HD = BASELINE_MODEL_DIMENSION // BASELINE_MODEL_NUMBER_OF_HEADS
        # 1. Pad to MAX_LEN
        inputs = torch.cat([value, torch.zeros(pad_shape(value), device = self.device)], dim = 1)
        inputs_shape = inputs.shape
        mask = torch.squeeze(torch.squeeze(mask, dim = 1), dim = 1)
        mask = torch.cat([mask, torch.zeros(pad_shape(mask, masks = True), device = self.device, dtype=torch.bool) ], dim = 1)
        # 2. Flatten
        mask = torch.repeat_interleave(mask, inputs.shape[-1] ,dim=1)
        inputs = inputs.reshape((inputs_shape[0], inputs_shape[1]* inputs_shape[2]))
        # 3. Compute
        inputs = inputs*mask
        # mask = mask.reshape((B,MAX_LEN  * HD, BASELINE_MODEL_NUMBER_OF_HEADS)).transpose(1,2)
        outputs = self.ff(inputs, mask)
        # 4. Unflatten and unpad
        # shape = BxNHxSxHD
        outputs = outputs.reshape((outputs.shape[0], MAX_LEN, -1, HD)).transpose(1,2)
        outputs, padding =  torch.split(outputs,[S, MAX_LEN - S] , dim = 2)   
        assert(np.prod(padding.shape) == (padding == 0).sum())
        return outputs 

def replace_mha(transformer: nn.Module, substitute: nn.Module, layer:int, device = "cuda", attention_type = "encoder"):
    if attention_type == "encoder":
        if not type(transformer.encoder.encoder_layers[layer].multi_headed_attention) == MultiHeadedAttention2:
            raise TypeError("Use function mha_to_mha2 first")
        transformer.encoder.encoder_layers[layer].multi_headed_attention.attention = AttentionSubstitute(substitute, device = device)  
        
    elif attention_type == "decoder_self": 
            transformer.decoder.decoder_layers[layer].trg_multi_headed_attention.attention =  AttentionSubstituteDecoder(substitute, device)
            
    elif attention_type == "decoder_cross":
        raise NotImplementedError()
    
    else:
        raise ValueError("attention_type must be in ['encoder', 'decoder_self', 'decoder_cross']")
    
    return transformer

class AttentionSubstituteDecoder(nn.Module):
    def __init__(self, FF_net:nn.Module, device = "cuda"):
        """Substitutes mha with a single FF. 

        Args:
            ff_list (): Feed forward nets that compute the attention values
        """
        super().__init__()
        self.ff = FF_net
        self.device = device
        
    def forward(self, query, key, value, mask):
        """This layer substitutes the Attention layer in mha2. 

        Args:
            query (Tensor): B x S x MD
            key (Tensor):  B x S x MD
            value (Tensor):  B x S x MD
            mask (Tensor): B x 1 x 1 x S

        Returns:
            Tensor: B x NH x S x HD
        """
        S = value.shape[1]
        B = len(value)
        HD = BASELINE_MODEL_DIMENSION // BASELINE_MODEL_NUMBER_OF_HEADS
        batch_size = value.shape[0]
        # 1. Pad to MAX_LEN
        inputs = torch.cat([value, torch.zeros(pad_shape(value), device = self.device)], dim = 1)
        mask = mask.squeeze(dim = 1)
        trg_padding_mask = pad_sequence([x[-1] for x in mask], batch_first=True, padding_value=0)
        trg_padding_mask = torch.cat([trg_padding_mask, torch.zeros(pad_shape(trg_padding_mask, masks = True), dtype=torch.bool, device = self.device)], dim = 1).view(batch_size, 1, -1)
         
        trg_no_look_forward_mask = torch.triu(torch.ones((1, MAX_LEN, MAX_LEN), device=self.device) == 1).transpose(1, 2)

        # logic AND operation (both padding mask and no-look-forward must be true to attend to a certain target token)
        trg_mask = trg_padding_mask & trg_no_look_forward_mask  # final shape = (B, T, T)
        # 3. Compute
        outputs = self.ff(inputs, trg_mask)
        outputs = outputs.reshape((outputs.shape[0], MAX_LEN, -1, HD)).transpose(1,2)
        outputs, padding =  torch.split(outputs,[S, MAX_LEN - S] , dim = 2)   
        return outputs 

def substitute_ALR_encoder(baseline_transformer, substitute_class, substitute_model_path, layers, epoch, untrained, multi_device):
    import models.definitions.ALR_FF as m
    FF_net = getattr(m, substitute_class)
    print(f"Substituing attention with {FF_net}")
    mha_to_mha2(baseline_transformer)
    layers = layers if layers is not None else range(6)    
    print(layers)
    for l in layers:
        ff_net = FF_net()
        if not multi_device:
            ff_net.to(device)
        if not untrained:
            model_path=os.path.join(substitute_model_path ,f'layer{l}', ALR_CHECKPOINT_FORMAT.format(epoch, l))
            print(f"Loading weights from {model_path}")
            model_state = torch.load(model_path)
            ff_net.load_state_dict(model_state)
        else:
            print("Test uninitialized")
        ff_net.eval()
        replace_mha(baseline_transformer, ff_net, l, device)

def substitute_ALR_decoder(baseline_transformer, substitute_class, substitute_model_path, layers, epoch, untrained, multi_device):
    import models.definitions.ALR_FF as m
    FF_net = getattr(m, substitute_class)
    print(f"Substituing attention with {FF_net}")
    mha_to_mha2(baseline_transformer, attention_type="decoder_self")
    layers = layers if layers is not None else range(6)    
    print(layers)
    for l in layers:
        ff_net = FF_net()
        if not multi_device:
            ff_net.to(device)
        if not untrained:
            model_path=os.path.join(substitute_model_path, f'layer{l}', ALR_CHECKPOINT_FORMAT.format(epoch, l))
            print(f"Loading weights from {model_path}")
            model_state = torch.load(model_path)
            ff_net.load_state_dict(model_state)
        else:
            print("Test uninitialized")
        ff_net.eval()
        replace_mha(baseline_transformer, ff_net, l, device, attention_type="decoder_self")

def substitute_separate_mha(baseline_transformer, substitute_class, substitute_model_path, layers, epoch, untrained):
    import models.definitions.mha_FF as m
    FF_net =getattr(m, substitute_class)
    print(f"Substituing attention with {FF_net}")
    mha_to_mha2(baseline_transformer)
    layers = layers if layers is not None else range(6)    
    print(layers)
    for l in layers:
        ff_nets=[]
        for h in range(8):
            ckpt_model_name = MHA_SEPARATE_CHECKPOINT_FORMAT.format(epoch,l, h)
            ff_net = FF_net().to(device)
            if not untrained:
                model_path=os.path.join(substitute_model_path,f'layer{l}' ,ckpt_model_name) 
                model_state = torch.load(model_path)
                ff_net.load_state_dict(model_state)
                ff_net.eval()
            else:
                ff_net.train()
            ff_nets+=[ff_net]
        replace_mha_separate_heads(baseline_transformer, ff_nets, l, device)
      
def substitute_sublayer(baseline_transformer, substitute_class, substitute_model_path, layers, epoch, untrained):
    import models.definitions.ALRR_FF as m
    FF_net =getattr(m, substitute_class)
    print(f"Substituing attention with {FF_net}")
    mha_to_mha2(baseline_transformer)
    layers = layers if layers is not None else range(6)
    print(layers)
    # Step 3: Substitute attention layers   
    for l in layers:
        ff_net = FF_net().to(device)
        if not untrained:
            ckpt_model_name = ALR_CHECKPOINT_FORMAT.format(epoch, l)
            model_path=os.path.join(substitute_model_path, 'layer{0}'.format(l), ckpt_model_name)
            model_state = torch.load(model_path)
            ff_net.load_state_dict(model_state)
            ff_net.eval()
        else:
            ff_net.train()
        replace_sublayer(baseline_transformer, ff_net, l, device)
    
def substitute_attention(baseline_transformer, substitute_class, substitute_model_path, layer, epoch,t, untrained=False,  multi_device = False, decoder = False):
    
    if t == "ALR":
        print("Substitute mha only")
        if decoder == False:
            substitute_ALR_encoder(baseline_transformer, substitute_class, substitute_model_path, layer, epoch, untrained, multi_device)
        else:
            substitute_ALR_decoder(baseline_transformer, substitute_class, substitute_model_path, layer, epoch, untrained, multi_device)
    elif t == "ALRR":
        print("Substitute ELR layer")
        substitute_sublayer(baseline_transformer, substitute_class, substitute_model_path, layer, epoch, untrained)
    elif t == "mha_separate_heads":
        print("Substitute separate mha layer")
        substitute_separate_mha(baseline_transformer, substitute_class, substitute_model_path, layer, epoch, untrained)
    else:
        raise ValueError("Attention type in ['ALR', 'ALRR', 'mha_separate_heads']")
def pad_shape(batch, masks = False):
    shape = batch.shape
    if masks:
        return shape[0],MAX_LEN-shape[1] 
    return shape[0], MAX_LEN-shape[1], shape[2]
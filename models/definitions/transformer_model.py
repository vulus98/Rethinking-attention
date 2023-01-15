"""
    Contains the implementation of the original transformer paper "Attention is all you need".

    Paper link: https://arxiv.org/pdf/1706.03762.pdf

    Certain modifications:
    1. LayerNorm (before instead of after)
    2. Dropout (Added additionally to attention weights and point-wise feed-forward net sublayer

    Suggested theory: https://jalammar.github.io/illustrated-transformer/ (amazing blog!)

"""


import math
import copy


import torch
import numpy as np
import torch.nn as nn
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence


from utils.constants import *


class Transformer(nn.Module):

    def __init__(self, model_dimension, src_vocab_size, trg_vocab_size, number_of_heads, number_of_layers, dropout_probability, log_attention_weights=False):
        super().__init__()

        # Embeds source/target token ids into embedding vectors
        self.src_embedding = Embedding(src_vocab_size, model_dimension)
        self.trg_embedding = Embedding(trg_vocab_size, model_dimension)

        # Adds positional information to source/target token's embedding vector
        # (otherwise we'd lose the positional information which is important in human languages)
        self.src_pos_embedding = PositionalEncoding(model_dimension, dropout_probability)
        self.trg_pos_embedding = PositionalEncoding(model_dimension, dropout_probability)

        # All of these will get deep-copied multiple times internally
        mha = MultiHeadedAttention(model_dimension, number_of_heads, dropout_probability, log_attention_weights)
        pwn = PositionwiseFeedForwardNet(model_dimension, dropout_probability)
        encoder_layer = EncoderLayer(model_dimension, dropout_probability, mha, pwn)
        decoder_layer = DecoderLayer(model_dimension, dropout_probability, mha, pwn)

        self.encoder = Encoder(encoder_layer, number_of_layers)
        self.decoder = Decoder(decoder_layer, number_of_layers)

        # Converts final target token representations into log probabilities vectors of the target vocab size
        self.decoder_generator = DecoderGenerator(model_dimension, trg_vocab_size)
        self.init_params()

    def init_params(self, default_initialization=False):
        # Not mentioned in the paper, but other implementations used xavier.
        # I tested both PyTorch's default initialization and this, and xavier has tremendous impact! I didn't expect
        # a model's perf, with normalization layers, to be so much dependent on the choice of weight initialization.
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, src_token_ids_batch, trg_token_ids_batch, src_mask, trg_mask):
        src_representations_batch = self.encode(src_token_ids_batch, src_mask)
        trg_log_probs = self.decode(trg_token_ids_batch, src_representations_batch, trg_mask, src_mask)
        return trg_log_probs

    # Modularized into encode/decode functions for optimizing the decoding/translation process (see translation script)
    def encode(self, src_token_ids_batch, src_mask):
        src_embeddings_batch = self.src_embedding(src_token_ids_batch)  # get embedding vectors for src token ids
        src_embeddings_batch = self.src_pos_embedding(src_embeddings_batch)  # add positional embedding
        src_representations_batch = self.encoder(src_embeddings_batch, src_mask)  # forward pass through the encoder

        return src_representations_batch

    def decode(self, trg_token_ids_batch, src_representations_batch, trg_mask, src_mask):
        trg_embeddings_batch = self.trg_embedding(trg_token_ids_batch)  # get embedding vectors for trg token ids
        trg_embeddings_batch = self.trg_pos_embedding(trg_embeddings_batch)  # add positional embedding
        # Shape (B, T, D), where B - batch size, T - longest target token-sequence length and D - model dimension
        trg_representations_batch = self.decoder(trg_embeddings_batch, src_representations_batch, trg_mask, src_mask)

        # After this line we'll have a shape (B, T, V), where V - target vocab size, decoder generator does a simple
        # linear projection followed by log softmax
        trg_log_probs = self.decoder_generator(trg_representations_batch)

        # Reshape into (B*T, V) as that's a suitable format for passing it into KL div loss
        trg_log_probs = trg_log_probs.reshape(-1, trg_log_probs.shape[-1])

        return trg_log_probs  # the reason I use log here is that PyTorch's nn.KLDivLoss expects log probabilities


#
# Encoder architecture
#


class Encoder(nn.Module):

    def __init__(self, encoder_layer, number_of_layers):
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer), f'Expected EncoderLayer got {type(encoder_layer)}.'

        self.encoder_layers = get_clones(encoder_layer, number_of_layers)
        self.norm = nn.LayerNorm(encoder_layer.model_dimension)

    def forward(self, src_embeddings_batch, src_mask):
        # Just update the naming so as to reflect the semantics of what this var will become (the initial encoder layer
        # has embedding vectors as input but later layers have richer token representations)
        src_representations_batch = src_embeddings_batch
        print(src_embeddings_batch.shape)
        # Forward pass through the encoder stack
        for encoder_layer in self.encoder_layers:
            # src_mask's role is to mask/ignore padded token representations in the multi-headed self-attention module
            src_representations_batch = encoder_layer(src_representations_batch, src_mask)

        # Not mentioned explicitly in the paper (a consequence of using LayerNorm before instead of after the sublayer
        # check out the SublayerLogic module)
        return self.norm(src_representations_batch)

class EncoderLayer(nn.Module):

    def __init__(self, model_dimension, dropout_probability, multi_headed_attention, pointwise_net):
        super().__init__()
        num_of_sublayers_encoder = 2
        self.sublayers = get_clones(SublayerLogic(model_dimension, dropout_probability), num_of_sublayers_encoder)

        self.multi_headed_attention = multi_headed_attention
        self.pointwise_net = pointwise_net

        self.model_dimension = model_dimension

    def forward(self, src_representations_batch, src_mask):
        # Define anonymous (lambda) function which only takes src_representations_batch (srb) as input,
        # this way we have a uniform interface for the sublayer logic.
        encoder_self_attention = lambda srb: self.multi_headed_attention(query=srb, key=srb, value=srb, mask=src_mask)

        # Self-attention MHA sublayer followed by point-wise feed forward net sublayer
        src_representations_batch = self.sublayers[0](src_representations_batch, encoder_self_attention)
        src_representations_batch = self.sublayers[1](src_representations_batch, self.pointwise_net)

        return src_representations_batch


#
# Decoder architecture
#


class Decoder(nn.Module):

    def __init__(self, decoder_layer, number_of_layers):
        super().__init__()
        assert isinstance(decoder_layer, DecoderLayer), f'Expected DecoderLayer got {type(decoder_layer)}.'

        self.decoder_layers = get_clones(decoder_layer, number_of_layers)
        self.norm = nn.LayerNorm(decoder_layer.model_dimension)

    def forward(self, trg_embeddings_batch, src_representations_batch, trg_mask, src_mask):
        # Just update the naming so as to reflect the semantics of what this var will become
        trg_representations_batch = trg_embeddings_batch

        # Forward pass through the decoder stack
        for decoder_layer in self.decoder_layers:
            # Target mask masks pad tokens as well as future tokens (current target token can't look forward)
            trg_representations_batch = decoder_layer(trg_representations_batch, src_representations_batch, trg_mask, src_mask)

        # Not mentioned explicitly in the paper (a consequence of using LayerNorm before instead of after the sublayer
        # check out the SublayerLogic module)
        return self.norm(trg_representations_batch)


class DecoderLayer(nn.Module):

    def __init__(self, model_dimension, dropout_probability, multi_headed_attention, pointwise_net):
        super().__init__()
        num_of_sublayers_decoder = 3
        self.sublayers = get_clones(SublayerLogic(model_dimension, dropout_probability), num_of_sublayers_decoder)

        self.trg_multi_headed_attention = copy.deepcopy(multi_headed_attention)
        self.src_multi_headed_attention = copy.deepcopy(multi_headed_attention)
        self.pointwise_net = pointwise_net

        self.model_dimension = model_dimension

    def forward(self, trg_representations_batch, src_representations_batch, trg_mask, src_mask):
        # Define anonymous (lambda) function which only takes trg_representations_batch (trb - funny name I know)
        # as input - this way we have a uniform interface for the sublayer logic.
        # The inputs which are not passed into lambdas are "cached" here that's why the thing works.
        srb = src_representations_batch  # simple/short alias
        decoder_trg_self_attention = lambda trb: self.trg_multi_headed_attention(query=trb, key=trb, value=trb, mask=trg_mask)
        decoder_src_attention = lambda trb: self.src_multi_headed_attention(query=trb, key=srb, value=srb, mask=src_mask)

        # Self-attention MHA sublayer followed by a source-attending MHA and point-wise feed forward net sublayer
        trg_representations_batch = self.sublayers[0](trg_representations_batch, decoder_trg_self_attention)
        trg_representations_batch = self.sublayers[1](trg_representations_batch, decoder_src_attention)
        trg_representations_batch = self.sublayers[2](trg_representations_batch, self.pointwise_net)

        return trg_representations_batch


#
# Helper modules (designed with modularity in mind) and organized top to bottom.
#


# Note: the original paper had LayerNorm AFTER the residual connection and addition operation
# multiple experiments I found showed that it's more effective to do it BEFORE, how did they figure out which one is
# better? Experiments! There is a similar thing in DCGAN and elsewhere.
class SublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, representations_batch, sublayer_module):
        # Residual connection between input and sublayer output, details: Page 7, Chapter 5.4 "Regularization",
        return representations_batch + self.dropout(sublayer_module(self.norm(representations_batch)))


class DecoderGenerator(nn.Module):
    def __init__(self, model_dimension, vocab_size):
        super().__init__()

        self.linear = nn.Linear(model_dimension, vocab_size)

        # -1 stands for apply the log-softmax along the last dimension i.e. over the vocab dimension as the output from
        # the linear layer has shape (B, T, V), B - batch size, T - max target token-sequence, V - target vocab size
        # again using log softmax as PyTorch's nn.KLDivLoss expects log probabilities (just a technical detail)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, trg_representations_batch):
        # Project from D (model dimension) into V (target vocab size) and apply the log softmax along V dimension
        return self.log_softmax(self.linear(trg_representations_batch))


class PositionwiseFeedForwardNet(nn.Module):
    """
        It's position-wise because this feed forward net will be independently applied to every token's representation.

        Representations batch is of the shape (batch size, max token sequence length, model dimension).
        This net will basically be applied independently to every token's representation (you can think of it as if
        there was a nested for-loop going over the batch size and max token sequence length dimensions
        and applied this net to token representations. PyTorch does this auto-magically behind the scenes.

    """
    def __init__(self, model_dimension, dropout_probability, width_mult=4):
        super().__init__()

        self.linear1 = nn.Linear(model_dimension, width_mult * model_dimension)
        self.linear2 = nn.Linear(width_mult * model_dimension, model_dimension)

        # This dropout layer is not explicitly mentioned in the paper but it's common to use to avoid over-fitting
        self.dropout = nn.Dropout(p=dropout_probability)
        self.relu = nn.ReLU()

    def forward(self, representations_batch):
        return self.linear2(self.dropout(self.relu(self.linear1(representations_batch))))


class MultiHeadedAttention(nn.Module):
    """
        This module already exists in PyTorch. The reason I implemented it here from scratch is that
        PyTorch implementation is super complicated as they made it as generic/robust as possible whereas
        on the other hand I only want to support a limited use-case.

        Also this is arguable the most important architectural component in the Transformer model.

        Additional note:
        This is conceptually super easy stuff. It's just that matrix implementation makes things a bit less intuitive.
        If you take your time and go through the code and figure out all of the dimensions + write stuff down on paper
        you'll understand everything. Also do check out this amazing blog for conceptual understanding:

        https://jalammar.github.io/illustrated-transformer/

        Optimization notes:

        qkv_nets could be replaced by Parameter(torch.empty(3 * model_dimension, model_dimension)) and one more matrix
        for bias, which would make the implementation a bit more optimized. For the sake of easier understanding though,
        I'm doing it like this - using 3 "feed forward nets" (without activation/identity hence the quotation marks).
        Conceptually both implementations are the same.

        PyTorch's query/key/value are of different shape namely (max token sequence length, batch size, model dimension)
        whereas I'm using (batch size, max token sequence length, model dimension) because it's easier to understand
        and consistent with computer vision apps (batch dimension is always first followed by the number of channels (C)
        and image's spatial dimensions height (H) and width (W) -> (B, C, H, W).

        This has an important optimization implication, they can reshape their matrix into (B*NH, S/T, HD)
        (where B - batch size, S/T - max src/trg sequence length, NH - number of heads, HD - head dimension)
        in a single step and I can only get to (B, NH, S/T, HD) in single step
        (I could call contiguous() followed by view but that's expensive as it would incur additional matrix copy)

    """

    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)  # identity activation hence "nets"
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        self.log_attention_weights = log_attention_weights  # should we log attention weights
        self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

    def attention(self, query, key, value, mask):
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

        # Step 4: Not defined in the original paper apply dropout to attention weights as well
        attention_weights = self.attention_dropout(attention_weights)

        # Step 5: based on attention weights calculate new token representations
        # attention_weights shape = (B, NH, S, S)/(B, NH, T, T) or (B, NH, T, S), value shape = (B, NH, S/T, HD)
        # Final shape (B, NH, S, HD) for source MHAs or (B, NH, T, HD) target MHAs (again MHAs are used in 3 contexts)
        intermediate_token_representations = torch.matmul(attention_weights, value)

        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        # Step 1: Input linear projection
        # Notation: B - batch size, NH - number of heads, S/T - max src/trg token-sequence length, HD - head dimension
        # Shape goes from (B, S/T, NH*HD) over (B, S/T, NH, HD) to (B, NH, S/T, HD) (NH*HD=D where D is model dimension)
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]

        # Step 2: Apply attention - compare query with key and use that to combine values (see the function for details)
        intermediate_token_representations, attention_weights = self.attention(query, key, value, mask)

        # Potentially, for visualization purposes, log the attention weights, turn off during training though!
        # I had memory problems when I leave this on by default
        if self.log_attention_weights:
            self.attention_weights = attention_weights

        # Step 3: Reshape from (B, NH, S/T, HD) over (B, S/T, NH, HD) (via transpose) into (B, S/T, NHxHD) which is
        # the same shape as in the beginning of this forward function i.e. input to MHA (multi-head attention) module
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1, self.number_of_heads * self.head_dimension)

        # Step 4: Output linear projection
        token_representations = self.out_projection_net(reshaped)

        return token_representations


#
# Input modules
#


class Embedding(nn.Module):

    def __init__(self, vocab_size, model_dimension):
        super().__init__()
        self.embeddings_table = nn.Embedding(vocab_size, model_dimension)
        self.model_dimension = model_dimension

    def forward(self, token_ids_batch):
        assert token_ids_batch.ndim == 2, f'Expected: (batch size, max token sequence length), got {token_ids_batch.shape}'

        # token_ids_batch has shape (B, S/T), where B - batch size, S/T max src/trg token-sequence length
        # Final shape will be (B, S/T, D) where D is the model dimension, every token id has associated vector
        embeddings = self.embeddings_table(token_ids_batch)

        # (stated in the paper) multiply the embedding weights by the square root of model dimension
        # Page 5, Chapter 3.4 "Embeddings and Softmax"
        return embeddings * math.sqrt(self.model_dimension)


class PositionalEncoding(nn.Module):

    def __init__(self, model_dimension, dropout_probability, expected_max_sequence_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)

        # (stated in the paper) Use sine functions whose frequencies form a geometric progression as position encodings,
        # (learning encodings will also work so feel free to change it!). Page 6, Chapter 3.5 "Positional Encoding"
        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension)

        # Checkout playground.py for visualization of how these look like (it's super simple don't get scared)
        positional_encodings_table = torch.zeros(expected_max_sequence_length, model_dimension)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies)  # sine on even positions
        positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies)  # cosine on odd positions

        # Register buffer because we want to save the positional encodings table inside state_dict even though
        # these are not trainable (not model's parameters) so they otherwise would be excluded from the state_dict
        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

        # embedding_batch's shape = (B, S/T, D), where S/T max src/trg token-sequence length, D - model dimension
        # So here we get (S/T, D) shape which will get broad-casted to (B, S/T, D) when we try and add it to embeddings
        positional_encodings = self.positional_encodings_table[:embeddings_batch.shape[1]]

        # (stated in the paper) Applying dropout to the sum of positional encodings and token embeddings
        # Page 7, Chapter 5.4 "Regularization"
        return self.dropout(embeddings_batch + positional_encodings)


#
# Helper model functions
#


def get_clones(module, num_of_deep_copies):
    # Create deep copies so that we can tweak each module's weights independently
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])


# Count how many trainable weights the model has <- just for having a feeling for how big the model is
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_state_dict_shapes_and_names(model):
    # This part helped me figure out that I don't have positional encodings saved in the state dict
    print(model.state_dict().keys())

    # This part helped me see that src MHA was missing in the decoder since both it and trg MHA were referencing
    # the same MHA object in memory - stupid mistake, happens all the time, embrace the suck!
    for name, param in model.named_parameters():
        print(name, param.shape)
        if not param.requires_grad:
            raise Exception('Expected all of the params to be trainable - no param freezing used.')


# Testing the correctness of the transformer model - feel free to ignore - I used it during model development
if __name__ == "__main__":
    use_big_transformer = False

    # Dummy data
    src_vocab_size = 11
    trg_vocab_size = 11
    src_token_ids_batch = torch.randint(1, 10, size=(3, 2))
    trg_token_ids_batch = torch.randint(1, 10, size=(3, 2))

    transformer = Transformer(
        model_dimension=BIG_MODEL_DIMENSION if use_big_transformer else BASELINE_MODEL_DIMENSION,
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        number_of_heads=BIG_MODEL_NUMBER_OF_HEADS if use_big_transformer else BASELINE_MODEL_NUMBER_OF_HEADS,
        number_of_layers=BIG_MODEL_NUMBER_OF_LAYERS if use_big_transformer else BASELINE_MODEL_NUMBER_OF_LAYERS,
        dropout_probability=BIG_MODEL_DROPOUT_PROB if use_big_transformer else BASELINE_MODEL_DROPOUT_PROB
    )

    # These 2 functions helped me figure out the 2 bugs I had:
    # 1) I did not register positional encodings and thus they wouldn't be saved and later model-loading would fail
    # 2) I had a bug with MHA (attention) in decoder, where both src and trg were referencing the same MHA object in mem
    # It's a good practice to see whether the names, shapes and number of params make sense.
    # e.g. I knew that the big transformer had ~175 M params and I verified that here.
    analyze_state_dict_shapes_and_names(transformer)
    print(f'Size of the {"big" if use_big_transformer else "baseline"} transformer = {count_parameters(transformer)}')

    out = transformer(src_token_ids_batch, trg_token_ids_batch, src_mask=None, trg_mask=None)

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

def pad_shape(batch, masks = False):
    shape = batch.shape
    if masks:
        return shape[0],MAX_LEN-shape[1] 
    return shape[0], MAX_LEN-shape[1], shape[2]

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


class MultiHeadedAttention3(nn.Module):
    def __init__(self, mha:MultiHeadedAttention):
            super().__init__()

            self.head_dimension = mha.head_dimension
            self.number_of_heads = mha.number_of_heads

            # Split mha into attention and linear layer in front. 
            self.attention_w = AttentionWeights( mha )
            self.out_projection_net = mha.out_projection_net
            self.V = mha.qkv_nets[2] # matrix to extract values
            
    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        
        # Extract attention weights and values
        attention_weights = self.attention_w(query, key, mask)
        value = self.V(value).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
        
        # Step 5: based on attention weights calculate new token representations
        # attention_weights shape = (B, NH, S, S)/(B, NH, T, T) or (B, NH, T, S), value shape = (B, NH, S/T, HD)
        # Final shape (B, NH, S, HD) for source MHAs or (B, NH, T, HD) target MHAs (again MHAs are used in 3 contexts)
        intermediate_token_representations = torch.matmul(attention_weights, value)
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1, self.number_of_heads * self.head_dimension)

        # Step 4: Output linear projection
        token_representations = self.out_projection_net(reshaped)

        return token_representations

def mha_to_mha3(transformer: Transformer, layers:list = [0,1,2,3,4,5]):
    """Substitutes MultiHeadedAttention with MultiHeadedAttention3. Useful to extract attention weights.

    Args:
        transformer (Transformer): _description_
        layers (list): list of layers to modify. By default all mha in the encoders are substituted
    """
    for l in layers:
        transformer.encoder.encoder_layers[l].multi_headed_attention =  MultiHeadedAttention3(transformer.encoder.encoder_layers[l].multi_headed_attention)

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

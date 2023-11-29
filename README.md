## The Rethinking-attention paper official implementation (AAAI24) :computer: = :rainbow:
This repository contains the code developed for the [Rethinking attention paper](https://arxiv.org/abs/2311.10642) which got accepted at AAAI24 conference.
In this project we tried to replace the self-attention with feed-forward networks (FFN) to evaluate the importance of attention inside the transformer.

The developed code builds on top of an open-source implementation by Aleksa Godric,
(:link: [ pytorch-original-transformer](https://github.com/gordicaleksa/pytorch-original-transformer)))
of the original transformer (:link: [Vaswani et al.](https://arxiv.org/abs/1706.03762)). <br/>.

## Table of Contents
  * [Environment setup](#environment-setup)
  * [Code overview](#code-overview)
  * [Train baseline transformer](#train-baseline-transformer)
  * [Intermediate data extraction](#intermediate-data-extraction)
  * [Train replacement FF networks](#full-sentence-approach)
  * [Evaluation](#evaluation)


## Environment setup

1. Navigate into the project directory `cd path_to_repo`
2. Run `conda env create` from project directory (this will create a brand new conda environment).
3. Run `rethinking-attention` (for running scripts from your console or set the interpreter in your IDE)
4. Run `export SCRATCH=path_to_outputs`, where `path_to_outputs` is the directory where you want the output files to be stored. If you are running this code on the Euler cluster, the variable is already defined.
5. Execute `./scripts/baseline/prepare_dataset.py` to download the IWSLT2017 dataset which will be used in this project. Choose between different subsets for download: English-to-German (E2G), English-to-French (E2F)...

In the following, all the commands are assumed to be run from the root of the repository.

## Code overview

As previously mentioned, the code was developed on top of an existing implementation of the transformer. Our main contribution to this code resides in
- the folder `./scripts` which contains scripts for extracting intermediate data, training different architectures and evaluating them in the transformer.
- The file `./utils/full_sentence_utils.py` contains the classes and functions for substituting FFN in the transformer.

The provided code was run on different GPUs all with a minimum of 11GB of memory, but up to 24GB.
In case your GPU does not have this much memory you should try to reduce the batch size. However, some of the bigger architectures may not work properly.

The description on how to run the code is general for any platform. Since we run the code on a cluster which uses slurm, we left in the `./submission_scripts` folder
the wrapper scripts which were used to submit jobs. If you want to use them, please either adjust the path output-path argument or create a folder `./sbatch_log`
which will collect all the program outputs.
The folder `./submission_scripts` is organised as the `./scripts` folder.
The wrapper script for *scripts/example_script.py* is located at *submission_scripts/example_script.sh*.

## Train baseline transformer

Initial step in the pipeline would be training the transformer. It could be done by running:

1. Execute `python3 ./scripts/baseline/training_script.py`

In the file you would be able to find specific arguments to control the training process.


## Intermediate data extraction

To train our FFNs we first extract the intermediate values that are given as input and output to the attention module. To extract the intermediate data (e.g. from English to German dataset) run
1. `python3 ./scripts/extraction/extract.py --path_to_weights ./models/binaries/Transformer_None_None_20.pth --batch_size 1400 --dataset_name IWSLT --language_direction en_de --model_name 128emb_20ep --output_path ./mha_outputs`
2. `python3 ./scripts/extraction/extract_mha.py --path_to_weights ./models/binaries/Transformer_None_None_20.pth --batch_size 1400 --dataset_name IWSLT --language_direction en_de --model_name 128emb_20ep --output_path ./mha_outputs`

The first script extracts inputs and outputs of
- each encoder layer (identified by *ELR* in the file name),
- each multi-headed attention (MHA) module (identified by *ALR* in the file name),
- each "sublayer zero" which consists of the MHA, the layer normalization and the residual connection (identified by *ALRR* in the file name).

The second script extracts inputs and outputs of 
- each MHA excluded the linear layer which mixes the values extracted by each head. This is to enable learning the output of each head separately as in the 'separate head' approach.

At the end of this section, your main folder should contain one folder *output_layers* containing the output of the first script and one folder *mha_outputs*
with the outputs of the second script.  This same script simultaneously extracts the data from all 3 types of attention and stores it in the same file. These values are used to train FFNs which replace attention with different layers of abstraction.

## Train replacement FF networks

In this step, the FFN takes in the concatenated word representations of a sentence as input and produces updated word representations as output in a single pass.
In order to handle input sentences of varying lengths, we have decided to pad all sentences to a maximum fixed length and mask the padded values with zeros to
prevent them from influencing the model's inference. 

We tried substituting attention with three layer of abstraction: 
- *Encoder Layer Replacement (ELR)*: replaces the whole encoder layer in the encoder
- *Attention Layer with Residual Replacement(ALRR)*: replaces the MHA and the residual connection
- *Attention Layer Replacement (ALR)*: replaces only the MHA
- *Attention Layer Separate heads Replacement (ALSR)*: replaces the same part as *ALR*, but one FFN is trained for each head.

The architecture used for each approach are listed in 
- `./models/definitions/ALRR_FF.py`
- `./models/definitions/ALR_FF.py`
- `./models/definitions/ALSR_FF.py`.
- `./models/definitions/ELR_FF.py`.

For the final experiment we considered 5 architectures ranging from extra small (XS)
to large (L). The considered range of number of parameters shows the operating range of the FFN. In particular, the XS network reduces the BLEU score of the transformer, while as the number of parameter grows, so does the BLEU up to saturation with the L network.
Each approach uses a different training script. Each training script contains a data loader responsible for loading the data extracted at the previous step and
creating batches of a fixed length *MAX_LEN* (using padding). Each training script receives as input the name of the substitute class (e.g. `FFNetwork_L`)
and the index of the layer to emulate. The training loop iterates over the training data for a specified maximum number of epochs.
The instruction for running the training scripts are listed below. 

### Training `ALRR`

To train one of the architectures defined in `models/definitions/ALRR.py` for a specific layer run:
`python3 ./scripts/full_sentence/training_ALRR.py --num_of_curr_trained_layer [0-5] --substitute_class <function name>`.
For example to train the network *FFNetwork_L* to substitute layer zero run
`python3 ./scripts/training_ALRR.py --num_of_curr_trained_layer 0 --substitute_class FFNetwork_L`.

### Training `ALSR`

To train one of the architectures defined in `./models/definitions/ALSR_FF.py` for a specific layer run:
`python3 ./scripts/full_sentence/training_ALR.py --num_of_curr_trained_layer [0-5] --substitute_class <function name>`.
For example to train the network *FFNetwork_L* to substitute layer zero with 8 heads, one for each head in the MHA of layer zero, run:
`python3 ./scripts/full_sentence/training_ALSR.py --num_of_curr_trained_layer 0 --substitute_class FFNetwork_L`.

### Training `ALRR`

To train one of the architectures defined in `models/definitions/ELR.py` for a specific layer run:
`python3 ./scripts/full_sentence/training_ELR.py --num_of_curr_trained_layer [0-5] --substitute_class <function name>`.
For example to train the network *FFNetwork_L* to substitute layer zero run
`python3 ./scripts/training_ELR.py --num_of_curr_trained_layer 0 --substitute_class FFNetwork_L`.


### Training `ALR`

To train one of the architectures defined in `./models/definitions/ALR_FF.py` for a specific layer run:
`python3 ./scripts/full_sentence/training_ALR.py --num_of_curr_trained_layer [0-5] --substitute_class <function name>`.
For example to train the network *FFNetwork_L* to substitute layer zero with 8 heads, one for each head in the MHA of layer zero, run:
`python3 ./scripts/full_sentence/training_ALR.py --num_of_curr_trained_layer 0 --substitute_class FFNetwork_L`.

### Training `ALR` in the decoder  
The `ALR` approach was also used to train self-attention and cross-attention in the decoder. The architecture used in the decoder are denoted by the word *decoder* and *cross_decoder* in the class name.
To train one of this architecture to substitute self-attention in the decoder layer run
`python3 ./scripts/full_sentence/training_ALR.py --num_of_curr_trained_layer [0-5] --substitute_class FFNetwork_decoder_L --decoder`

To train one of this architecture to substitute cross-attention in the decoder layer run
`python3 ./scripts/full_sentence/training_ALR.py --num_of_curr_trained_layer [0-5] --substitute_class FFNetwork_cross_decoder_L --decoder_ca`

In case you are running this code on a cluster which uses slurm, the script `./submission_scripts/training_ALR_FF_submit_all.sh` can be used to automatically
submit the training of a network for each layer (0-5).
If you use that script, please make sure that the path specified for the output of the program exists.
The script currently assumes a directory `./sbatch_log` which will collect all the outputs.

## Evaluation

All the networks trained in the previous step can be evaluated using `./scripts/full_sentence/validation_script.py`.
The validation is performed substituting the trained FFN in the pretrained transformer and computing the BLEU score on the validation data.
The script receives as inputs the following parameters: 
- `substitute_type`: type of approach to use for substitution. Must be in [`ALRR`, `ALR`, `ALSR`, `ELR`, `None`]. If `None`, no substitution takes place;
- `substitute_class`: class that substitutes attention e.g. *FFNetwork_L*;
- `layers`: list of layers to substitute. If layer is not specified, all layers are substituted;
- `epoch`: epoch checkpoint to use;

The second-to-last four attributes appended with `_d` can be used to substitute self-attention in the decoder, while last four, appended with `_d_ca` substitute cross-attention layers. Currently, only the `ALR` supports substitution
in the decoder layer.
To run the evaluation script the following command can be used
`python3 ./scripts/full_sentence/validation_script.py --substitute_type <subs_type> --substitute_class <class_name> --layers [0-5]* --epoch <epoch number>`
As an example if you want to evaluate the performance of *FFNetwork_L* in the `ALR` approach, substituting all layers in the encoder with
the checkpoint at epoch 21 the following command can be used:
`python3 ./scripts/full_sentence/validation_script.py --substitute_type ALR --substitute_class FFNetwork_L --epoch 21`

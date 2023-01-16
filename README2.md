## The Original Transformer (PyTorch) :computer: = :rainbow:
This repository contains the code developed for the Deep Learning project 2022.
In this project we tried to replace the self-attention with feed-forward networks (FFN) to evaluate the importance of attention inside the transformer.

The developed code builds on top of an open-source implementation by Aleksa Godric (who works for Deep Mind
(:link: [ pytorch-original-transformer](https://github.com/gordicaleksa/pytorch-original-transformer)))
of the original transformer (:link: [Vaswani et al.](https://arxiv.org/abs/1706.03762)). <br/>

## Table of Contents
  * [Environment setup](#environment-setup)
  * [Code overview](#code-overview)
  * [Train baseline transformer](#train-baseline-transformer)
  * [Intermediate data extraction](#intermediate-data-extraction)
  * [Full sentence approach](#full-sentence-approach)
  * [Average approach](#average-approach)
  * [Random feature appraoch](#random-feature-approach)

## Environment setup

1. Navigate into the project directory `cd path_to_repo`
2. Run `conda env create` from project directory (this will create a brand new conda environment).
3. Run `activate pytorch-transformer` (for running scripts from your console or set the interpreter in your IDE)
4. Run `export SCRATCH=path_to_outputs`, where `path_to_outputs` is the directory where you want the output files to be stored. If you are running this code on the euler cluster, the variable is already defined.
5. Execute `./scripts/download_iwslt.sh` to download the IWSLT dataset which will be used in this project.

In the following, all the commands are assumed to be run from the root of the repository.

## Code overview
As previously mentioned, the code was developed on top of an existing implementation of the transformer. Our main contribution to this code resides in
- the folder `scripts` which contains scripts for extracting intermediate data, training different architectures and evaluating them in the transformer.
- The file `simulator.py` which contains the classes and functions for substituting FFN in the transformer (for the average approach) with three different layers of abstraction: the entire encoder, the MHA and the residual connection and the MHA only.
- The file `full_sentence_utils` which contains the classes and functions for substituting FFN in the transformer (for the `full_sentence` approach).

The provided code was run on different GPUs all with a minimum of 11GB of memory, but up to 24GB.
In case your GPU does not have this much memory you should try to reduce the batch size. However, some of the bigger architecture may not work properly.

The description on how to run the code is general for any platform. Since we run the code on a cluster which uses slurm, we left in the submission_scripts folder the wrapper scripts which were used to submit jobs. If you want to use them, please either adjust the path output-path argument or create a folder `../sbatch_log` which will collect all the program outputs.
The folder submission_scripts is organised as the `scripts` folder. The wrapper script for *example_script.py* for the *exaple_approach* is located at *submission_scripts/example_approach/example_script.sh*.
## Train baseline transformer
Note: since we already provide the pretrained transformer this step can be skipped.
The weights of a pretrained transformer are saved in the directory `./models/binaries/transformer_128.pth`.
The following parts of the project will use this transformer to extract intermediate values which will be used to train the FFNs to replace attention blocks.
If you want to train this transformer yourself

1. Execute `python3 ./scripts/baseline/training_script.py`
2. Copy the checkpoint after 20 epochs executing `cp $SCRATCH/models/checkpoint/transformer_ckpt_epoch_20.pth models/binaries/transformer_128.pth`

## Intermediate data extraction

To train our FFNs we first extract the intermediate values that are given as input and output to the attention module. To extract the intermediate data run
1. `python3 scripts/extraction/extract.py --path_to_weights models/binaries/transformer_128.pth --batch_size 1400 --dataset_name IWSLT --language_direction E2G --model_name 128emb_20ep`
2. `python scripts/extraction/extract_mha.py --batch_size 1400 --dataset_name IWSLT --language_direction E2G --model_name 128emb_20ep --path_to_weights models/binaries/transformer_128.pth --output_path $SCRATCH/mha_outputs`

The first script extracts inputs and outputs of
- each encoder layer (identified by *ELR* in the file name),
- each multi-headed attention (MHA) module (identified by *ALR* in the file name),
- each "sublayer zero" which consists of the MHA, the layer normalization and the residual connection (identified by *ALRR* in the file name).

The second script extracts inputs and outputs of 
- each MHA excluded the linear layer which mixes the values extracted by each head. This is to enable learning the output of each head separately as in the 'separate head' approach.

At the end of this section, your SCRATCH folder should contain one folder *output_layers* containing the output of the first script and one folder *mha_outputs*
with the outputs of the second script. These values are used to train FFNs which replace attention with different layers of abstraction.

## Full sentence approach

In this approach, the FFN takes in the concatenated word representations of a sentence as input and produces updated word representations as output in a single pass.
In order to handle input sentences of varying lengths, we have decided to pad all sentences to a maximum fixed length and mask the padded values with zeros to
prevent them from influencing the model's inference. 

We tried substituting attention with three layer of abstraction: 
- *Attention Layer with Residual Replacement(ALRR)*: replaces the MHA and the residual connection
- *Attention Layer Replacement (ALR)*: replaces only the MHA
- *Attention Layer Separate heads Replacement (ALSR)*: replaces the same part as *ALR*, but one FFN is trained for each head.

The architecture used for each approach are listed in 
- `models/definitions/ALRR_FF.py`
- `models/definitions/ALR_FF.py`
- `models/definitions/ALSR_FF.py`.

For the final experiment we considered 5 architectures ranging from extra small (XS)
to extra large (XL). The considered range of number of parameters shows the operating range of the FFN. In particular, the XS network reduces the BLEU score of the transformer, while as the number of parameter grows, so does the BLEU up to saturation with the XL network.
Each approach uses a different training script. Each training script contains a data loader responsible for loading the data extracted at the previous step and
creating batches of a fixed length *MAX_LEN* (using padding). Each training script receives as input the name of the substitute class (e.g. `FFNetwork_L`)
and the index of the layer to emulate. The training loop iterates over the training data for a specified maximum number of epochs.
The instruction for running the training scripts are listed below. 

### Training `ALRR`
To train one of the architectures defined in `models/definitions/ALRR.py` for a specific layer run:
`python3 scripts/full_sentence/training_ALRR.py --num_of_curr_trained_layer [0-5] --substitute_class <function name>`.
For example to train the network *FFNetwork_L* to substitute layer zero run
`python3 scripts/training_ALRR.py --num_of_curr_trained_layer 0 --substitute_class FFNetwork_L`.

### Training `ALSR`
To train one of the architectures defined in `models/definitions/ALSR_FF.py` for a specific layer run:
`python3 scripts/full_sentence/training_ALR.py --num_of_curr_trained_layer [0-5] --substitute_class <function name>`.
For example to train the network *FFNetwork_L* to substitute layer zero with 8 heads, one for each head in the MHA of layer zero, run:
`python3 scripts/full_sentence/training_ALSR.py --num_of_curr_trained_layer 0 --substitute_class FFNetwork_L`.

### Training `ALR`
To train one of the architectures defined in `models/definitions/ALR_FF.py` for a specific layer run:
`python3 scripts/full_sentence/training_ALR.py --num_of_curr_trained_layer [0-5] --substitute_class <function name>`.
For example to train the network *FFNetwork_L* to substitute layer zero with 8 heads, one for each head in the MHA of layer zero, run:
`python3 scripts/full_sentence/training_ALR.py --num_of_curr_trained_layer 0 --substitute_class FFNetwork_L`.
This approach was also used to train self-attention in the decoder. The architecture used in the decoder are denoted by the word *decoder* in the class name.
To train one of this architecture to substitute self-attention in the encoder layer run
`python3 .scripts/full_sentence/training_ALR.py --num_of_curr_trained_layer [0-5] --substitute_class FFNetwork_decoder_L --decoder`

In case you are running this code on a cluster which uses slurm, the script `submission_scripts/training_ALR_FF_submit_all.sh` can be used to automatically
submit the training of a network for each layer (0-5).
If you use that script, please make sure that the path specified for the output of the program exists.
The script currently assumes a directory `../sbatch_log` which will collect all the outputs.

### Evaluation

All the networks trained in the previous step can be evaluated using `scripts/full_sentence/validation_script.py`.
The validation is performed substituting the trained FFN in the pretrained transformer and computing the BLUE score on the validation data.
The script receives as inputs the following parameters: 
- `substitute_type`: type of approach to use for substitution. Must be in [`ALRR`, `ALR`, `ALSR`, `none`]. If `none`, no substitution takes place;
- `substitute_class`: class that substitutes attention e.g. *FFNetwork_L*;
- `layers`: list of layers to substitute. If layer is not specified, all layers are substituted;
- `epoch`: epoch checkpoint to use;
- `untrained`: bool. If set, the substitute FF is not loaded with the trained weights and it is left untrained. This can be set to test the performance of a randomly substituted FFN.

The last four attributes appended with `_d` can be used to substitute self-attention in the decoder. Currently, only the `ALR` supports substitution
in the decoder layer.
To run the evaluation script the following command can be used
`python3 scripts/full_sentence/validation_script.py --substitute_type <subs_type> --substitute_class <class_name> --layers [0-5]* --epoch <epoch number>`
As an example if you want to evaluate the performance of *FFNetwork_L* in the `ALR` approach, substituting all layers in the encoder with
the checkpoint at epoch 21 the following command can be used:
`python3 scripts/full_sentence/validation_script.py --substitute_type ALR --substitute_class FFNetwork_L --epoch 21`

## Average approach

In this approach, the FFN takes in the concatenation of a word representation and the average of the representation of all the words in the sentence and produces the updated word representations as output. This is done for all words in each sentence. The idea behind this approach is to understand if the average of the word representation is enough information to learn the next representation. Moreover, the advantage of this approach is that it is not dependent on the sentence length.

Similarly to the full sentence approach, the average approach tries to replace the attention with three level of abstraction:
- The entire encoder layer (referenced as `ELR`).
- The MHA only (referenced as `ALR`).
- The MHA with the residual connection (referenced as `ALRR`).

Moreover, it was tested whether the whole encoder could be replaced by a single network transforming words using the sentence average as explained above.
This means that no intermediate layer activations are taken into consideration and the whole structure of the encoder was ignored.

If you have access to a cluster with slurm running, there are some convenience scripts in the `submission_scripts` folder.

### Architectures exploration

The optuna package was used to perform a randomized grid search with a manually defined search spaceto find a good architecture that
could substitute the attention blocks for all three approaches of layer replacement (`ALR`, `ALRR`, `ELR`) as well
as for the replacement of the whole encoder.
The best performing architectures are used in the following steps to train FFN that simulate the substituted blocks.

To run the randomized grid search for the layer replacement, run:
`python3 scripts/averaging/find_single_layer_arch.py --[ELR|ALR|ALRR] --input <index-input> --output <index-output>`.

In the command index-input and index-output are the indexes identifying the input and output layer considered for the search.

In practice, we set index-input = index-output = 0 and use for all layers the architecture that performed best in layer 0.
We found in our experiments that the first layer is the hardest to learn and this approach works well in practice.


To start a randomized grid search to replace the whole encoder, run:
`python3 scripts/averaging/single_sim.py --input 0 --output norm`

This will search for a good architecture for a network replacing the whole encoder as it takes the input of encoder layer 0 as input and
the output of the layer normalization after the last encoder layer as its labels for train.


In all cases, the architectures found have to be manually put into the `configs` data-structure for evaluation (in `evaluate.py`)
or for further processing (in `sim_all_petrain.py` and `sim_all_together.py`).

### Training

After having put the found configurations in `sim_all_pretrain.py` this script will handle the training of all the layers for the three different approaches.
The training can be run with the following command
`python3 ./scripts/averaging/sim_all_pretrain.py --[ELR|ALR|ALRR]`. 

For the approach replacing the whole encoder, no further training is required, as this is already done in `single_sim.py` when searching for a good architecture.

### Evaluation

For evaluation, the script `evaluate.py` can be used. To run the script use the command:
`python3 ./scripts/averaging/evaluate.py --[ELR|ALR|ALRR|single_sim|vanilla]`

This will compute the BLEU score for the specified approach (or the original "vanilla" transformer) using the pretrained networks.
Moreover, it will try to fine tune the pretrained networks in the whole transformer and compute the so obtained BLEU score.

Finally, it also trains the whole transformer with the replaced module from scratch (both the transformer as well as the
new module will have their weights resetted) and again computes the corresponding BLEU score to see whether the pre-training is necessary.
### Random matrices
In this approach we wanted to test if randmoly initialized matrices are sufficient to achieve a good BLEU score. The idea is that these matrices extract features of the input representation and as long as the outputs are correlated, attention may still work. The script `scripts/random_embedding/train_random_embedding.py` substitute all qkv matrices with randomly initialized matrices. All the matrices weights are blocked, while the rest of the transformer is trained. To run this experiment execute:
1. `python3 scripts/random_embedding/train_random_embedding.py`: this will replace all qkv nets with the same randomly initialized matrix.
2. `python3 scripts/random_embedding/train_random_embedding.py --independent_init`: this will initialize all three matrices independently.
The BLEU score is reported at the end of every epoch.

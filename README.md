<h1 align="center">
  <img src="doc/logo.png" width="500">
</h1>

**Paraplume** is a sequence-based paratope prediction method. It predicts which amino acids in an antibody sequence are likely to interact with an antigen during binding. Concretely, given an amino acid sequence, the model returns a **probability for each residue** indicating the likelihood of antigen interaction.

<h1 align="center">
  <img src="doc/example_github.png" width="500">
</h1>

<hr style="height:3px;border:none;background-color:#ff6b6b;" />


<details>
<summary><h1>📖 HOW IT WORKS</h1></summary>

Paraplume uses supervised learning and involves three main steps:

1. **Labelling**:
   Antibody sequences are annotated with paratope labels using structural data from [SAbDab](http://opig.stats.ox.ac.uk/webapps/newsabdab/).

2. **Sequence representation**:
   Each amino acid is embedded into a high-dimensional vector using **Protein Language Model (PLM) embeddings**.

3. **Model training**:
   A **Multi-Layer Perceptron (MLP)** is trained to minimize **Binary Cross-Entropy Loss**, using PLM embeddings as inputs and paratope labels as targets.

The full workflow of Paraplume is summarized Figure B below:

![Summary](./doc/figure1.png)

</details>

<hr style="height:3px;border:none;background-color:#ff6b6b;" />


<details>
<summary><h1>⚙️ INSTALLATION</h1></summary>

It is available on PyPI and can be installed through pip.

```bash
pip install paraplume
```

We recommend installing it in a virtual environment with python >= 3.10.

</details>

<hr style="height:3px;border:none;background-color:#ff6b6b;" />


<details>
<summary><h1>💻 COMMAND LINE INFERENCE </h1></summary>

A command-line tool for predicting paratopes from antibody sequences.
`infer-paratope` provides two commands, one to infer the paratope from a unique sequence (`seq-to-paratope`) and another from a batch of sequences in the form of a csv file (`file-to-paratope`).
```bash
infer-paratope COMMAND [OPTIONS][ARGS] ...
```
Run `infer-paratope --help` to see the commands.

<details>
<summary><h2>📋 Commands</h2></summary>

<details>
<summary><h3>1. seq-to-paratope - Predict from sequence</h3></summary>

Predict paratope directly from amino acid sequences provided as command line arguments.

#### Usage
```bash
infer-paratope seq-to-paratope [OPTIONS]
```

#### Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-h, --heavy-chain` | TEXT | - | Heavy chain amino acid sequence |
| `-l, --light-chain` | TEXT | - | Light chain amino acid sequence |
| `--custom-model` | PATH | None | Path to custom trained model folder |
| `--gpu` | INT | 0 | GPU device to use |
| `--large/--small` | flag | --large | Model size (large: full Paraplume, small: ESM-2 only) |

<details>
<summary><h4>Examples</h4></summary>

**Both chains:**
```bash
infer-paratope seq-to-paratope \
  -h QAYLQQSGAELVKPGASVKMSCKASDYTFTNYNMHWIKQTPGQGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCASLGSSYFDYWGQGTTLTVSS \
  -l EIVLTQSPTTMAASPGEKITITCSARSSISSNYLHWYQQKPGFSPKLLIYRTSNLASGVPSRFSGSGSGTSYSLTIGTMEAEDVATYYCHQGSNLPFTFGSGTKLEIK
```

**Heavy chain only:**
```bash
infer-paratope seq-to-paratope \
  -h QAYLQQSGAELVKPGASVKMSCKASDYTFTNYNMHWIKQTPGQGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCASLGSSYFDYWGQGTTLTVSS
```

**Light chain only:**
```bash
infer-paratope seq-to-paratope \
  -l EIVLTQSPTTMAASPGEKITITCSARSSISSNYLHWYQQKPGFSPKLLIYRTSNLASGVPSRFSGSGSGTSYSLTIGTMEAEDVATYYCHQGSNLPFTFGSGTKLEIK
```

</details>

</details>

<details>
<summary><h3>2. file-to-paratope - Predict from File</h3></summary>

Predict paratope from sequences stored in a CSV file.

#### Usage
```bash
infer-paratope file-to-paratope [OPTIONS] FILE_PATH
```

#### Arguments
| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `FILE_PATH` | PATH | ✓ | Path to input CSV file |

#### Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--custom-model` | PATH | None | Path to custom trained model folder |
| `--name` | TEXT | paratope_ | Prefix for output file |
| `--gpu` | INT | 0 | GPU device to use |
| `--emb-proc-size` | INT | 100 | Embedding batch size for memory management |
| `--compute-sequence-embeddings` | flag | False | Compute paratope and classical sequence embeddings |
| `--single-chain` | flag | False | Process single chain sequences |
| `--large/--small` | flag | --large | Model size (large: Paraplume, small: Paraplume-S, using ESM-2 embedding only) |



<details>
<summary><h4>Examples</h4></summary>

**Paired chains:**
```bash
infer-paratope file-to-paratope test.csv
```

**Heavy chain only:**
```bash
infer-paratope file-to-paratope test_heavy.csv --single-chain
```

**Light chain only:**
```bash
infer-paratope file-to-paratope test_light.csv --single-chain
```

Sample input files are available in `tests/data/`:
- `test.csv` - Paired heavy/light chains
- `test_heavy.csv` - Heavy chain only
- `test_light.csv` - Light chain only

</details>

<details>
<summary><h4>Input</h4></summary>

Your CSV file must contain these columns:

**For paired chains (default):**
| sequence_heavy | sequence_light |
|----------------|----------------|
| QAYLQQSGAELVKPGASVKMSCKASDYTFTNYNMHWIKQTPGQGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCASLGSSYFDYWGQGTTLTVSS | EIVLTQSPTTMAASPGEKITITCSARSSISSNYLHWYQQKPGFSPKLLIYRTSNLASGVPSRFSGSGSGTSYSLTIGTMEAEDVATYYCHQGSNLPFTFGSGTKLEIK |
| EVQLVESGGGLVQPGGSLRLSCAASGFTFSRYAMSWVRQAPGKGLEWVSVISSGGSYTYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKDREYRYYYYGMDVWGQGTTVTVSS | DIQMTQSPSSLSASVGDRVTITCRASQGISSWLAWYQQKPGKAPKLLIYDASSLESGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYGSSPPYTFGQGTKLEIK |

**For single heavy chain (use `--single-chain`):**
| sequence_heavy | sequence_light |
|----------------|----------------|
| QAYLQQSGAELVKPGASVKMSCKASDYTFTNYNMHWIKQTPGQGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCASLGSSYFDYWGQGTTLTVSS | |
| EVQLVESGGGLVQPGGSLRLSCAASGFTFSRYAMSWVRQAPGKGLEWVSVISSGGSYTYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKDREYRYYYYGMDVWGQGTTVTVSS | |

**For single light chain (use `--single-chain`):**
| sequence_heavy | sequence_light |
|----------------|----------------|
| | EIVLTQSPTTMAASPGEKITITCSARSSISSNYLHWYQQKPGFSPKLLIYRTSNLASGVPSRFSGSGSGTSYSLTIGTMEAEDVATYYCHQGSNLPFTFGSGTKLEIK |
| | DIQMTQSPSSLSASVGDRVTITCRASQGISSWLAWYQQKPGKAPKLLIYDASSLESGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYGSSPPYTFGQGTKLEIK |

</details>

<details>
<summary><h4>Output</h4></summary>

Creates a pickle file (e.g., `paratope_test.pkl`) containing:
- `model_prediction_heavy` - Paratope predictions for heavy chains
- `model_prediction_light` - Paratope predictions for light chains

**Reading results:**
```python
import pandas as pd
predictions = pd.read_pickle("paratope_test.pkl")
print(predictions.head())
```

</details>

</details>

</details>

</details>

<hr style="height:3px;border:none;background-color:#ff6b6b;" />

<details>
<summary><h1>🦮 COMMAND LINE TRAINING</h1></summary>

We also provide the possibility to use your custom model for inference. To train your custom model you will need to run two commands: `create-dataset` to generate labels and PLM embeddings for your desired dataset, and `train-model` to train the model.

After training the model on your custom dataset, the model is saved in a folder whose path can be given to the inference commands as a `--custom-model` option.

<details>
<summary><h2>📋 Commands</h2></summary>

<details>
<summary><h3>1. create-dataset - Generate Training Dataset</h3></summary>

Create dataset to train the neural network. Sequences and labels are saved in a .json file, and LPLM embeddings are saved in a .pt file.

#### Usage
```bash
create-dataset [OPTIONS] CSV_FILE_PATH PDB_FOLDER_PATH
```

#### Arguments
| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `CSV_FILE_PATH` | PATH | ✓ | Path of csv file to use for pdb list |
| `PDB_FOLDER_PATH` | PATH | ✓ | Pdb folder path for ground truth labeling |

#### Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--result-folder, -r` | PATH | result | Where to save results |
| `--emb-proc-size` | INTEGER | 100 | We create embeddings chunk by chunk to avoid memory explosion. This is the chunk size. Optimal value depends on your computer |
| `--gpu` | INTEGER | 0 | Which gpu to use |
| `--single-chain` | flag | False | Generate embeddings using llms on single chain mode, which slightly increases performance |

<details>
<summary><h4>Example</h4></summary>

```bash
create-dataset custom_train_set.csv pdb_folder \
  -r training_data \
  --gpu 0 \
  --emb-proc-size 50 \
  --single-chain
```

</details>

<details>
<summary><h4>Input</h4></summary>

`custom_train_set.csv` contains information about the PDB files used for training and has the following format:

| pdb  | Lchain | Hchain | antigen_chain |
|------|--------|--------|---------------|
| 1ahw | D      | E      | F             |
| 1bj1 | L      | H      | W             |
| 1ce1 | L      | H      | P             |

**Column descriptions:**
- `pdb`: PDB code of the antibody-antigen complex (should be available in `pdb_folder` as `pdb_folder/pdb_code.pdb`)
- `Lchain`: Light chain identifier used to label the paratope
- `Hchain`: Heavy chain identifier used to label the paratope
- `antigen_chain`: Antigen chain identifier used to label the paratope

</details>

<details>
<summary><h4>Output</h4></summary>

Creates a folder with the same name `custom_train_set` inside `training_data`, in which there are two files, `json.dict` with the sequences and labels, and `embeddings.pt` for the PLM embeddings.

</details>

</details>

<details>
<summary><h3>2. train-model - Train Neural Network</h3></summary>

Train the model given provided parameters and data.

#### Usage
```bash
train-model [OPTIONS] TRAIN_FOLDER_PATH VAL_FOLDER_PATH
```

#### Arguments
| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `TRAIN_FOLDER_PATH` | PATH | ✓ | Path of train folder |
| `VAL_FOLDER_PATH` | PATH | ✓ | Path of val folder |

#### Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--lr` | FLOAT | 0.001 | Learning rate to use for training |
| `--n_epochs, -n` | INTEGER | 1 | Number of epochs to use for training |
| `--result_folder, -r` | PATH | result | Where to save results |
| `--pos-weight` | FLOAT | 1 | Weight to give to positive labels |
| `--batch-size, -bs` | INTEGER | 10 | Batch size |
| `--mask-prob` | FLOAT | 0 | Probability with which to mask each embedding coefficient |
| `--dropouts` | TEXT | 0 | Dropout probabilities for each hidden layer, separated by commas. Example '0.3,0.3' |
| `--dims` | TEXT | 1000 | Dimensions of hidden layers. Separated by commas. Example '100,100' |
| `--override` | flag | False | Override results |
| `--seed` | INTEGER | 0 | Seed to use for training |
| `--l2-pen` | FLOAT | 0 | L2 penalty to use for the model weights |
| `--alphas` | TEXT | - | Whether to use different alphas labels to help main label |
| `--patience` | INTEGER | 0 | Patience to use for early stopping. 0 means no early stopping |
| `--emb-models` | TEXT | all | LLM embedding models to use, separated by commas. LLMs should be in 'ablang2','igbert','igT5','esm','antiberty','prot-t5','all'. Example 'igT5,esm' |
| `--gpu` | INTEGER | 0 | Which GPU to use |

<details>
<summary><h4>Example</h4></summary>

```bash
train-model training_data/custom_train_set training_data/custom_val_set \
  --lr 0.001 \
  -n 50 \
  -r training_results \
  --batch-size 32 \
  --dims 512,256 \
  --dropouts 0.2,0.1 \
  --patience 5 \
  --emb-models igT5,esm \
  --gpu 0
```

</details>

<details>
<summary><h4>Input</h4></summary>

The two arguments (`training_data/custom_train_set` and `training_data/custom_val_set` in the example) are paths of folders created by the previous `create-dataset` command.

</details>

<details>
<summary><h4>Output</h4></summary>

Model weights and training parameters are saved in a folder (`training_results` in the example, `results` by default).

</details>

**The resulting trained model can then be used at inference by passing the output folder path as the --custom-model argument of the inference commands (see inference command lines).**


</details>

</details>

</details>
<hr style="height:3px;border:none;background-color:#ff6b6b;" />
<details>
<summary><h1>🚀 TRAINING AND USING YOUR CUSTOM MODEL </h1></summary>

Here's a complete example of how to train and use your custom model:

## Set up
- Clone repository
- Make sure you are in `Paraplume`.
- Download PDB files from [SabDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/about#formats) using IMGT format and save them in `./all_structures/imgt`.

## Step 1: Create training and validation datasets from CSVs
```bash
create-dataset ./tutorial/custom_train_set.csv ./all_structures/imgt -r custom_folder
```
The folder `custom_folder` will be created. Inside this folder the folder `custom_train_set` is created in which there are two files, `dict.json` for the sequences and labels, and `embeddings.pt` for the PLM embeddings.
Repeat for the validation set (used for early stopping):
```bash
create-dataset ./tutorial/custom_val_set.csv ./all_structures/imgt -r custom_folder
```


## Step 2: Train the model
```bash
train-model ./custom_folder/custom_train_set ./custom_folder/custom_val_set \
  --lr 0.001 \
  -n 50 \
  --batch-size 8 \
  --dims 512,256 \
  --dropouts 0.2,0.1 \
  --patience 5 \
  --emb-models igT5,esm \
  --gpu 0 \
  -r ./custom_folder
```
This will save training results in `custom_folder`.
`checkpoint.pt` contains the weights of the model, `summary_dict.json` contains the parameters used for training, and `summary_plot.png` some plots showing the training process.

## Step 3: Use the trained custom model for inference
After training, your custom model will be saved in the results folder and can be used with inference commands using the `--custom-model` option.

```bash
infer-paratope file-to-paratope ./Paraplume/tutorial/paired.csv --custom-model ./custom_folder
```

And the result is available as `paratope_paired.pkl` !!

</details>
<hr style="height:3px;border:none;background-color:#ff6b6b;" />

<details>
<summary><h1>🐍 PYTHON TUTORIAL</h1></summary>

A python tutorial is available in the `tutorial` folder.

</details>

<hr style="height:3px;border:none;background-color:#ff6b6b;" />

# ⚡ QUICK START

1. **Install**: `pip install paraplume`
2. **Single sequence**: `infer-paratope seq-to-paratope -h YOUR_HEAVY_CHAIN -l YOUR_LIGHT_CHAIN`
3. **File batch**: `infer-paratope file-to-paratope your_file.csv`

For detailed usage, expand the sections above! 👆

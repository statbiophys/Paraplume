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

### Basic Usage
```bash
infer-paratope [OPTIONS] COMMAND [COMMAND OPTIONS][COMMAND ARGS] ...
```

### Global Options
`--help` Show help message and exit

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
<summary><h4>Input File Format</h4></summary>

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
<summary><h1>🐍 PYTHON TUTORIAL</h1></summary>

A python tutorial is available in the `tutorial` folder.

</details>

<hr style="height:3px;border:none;background-color:#ff6b6b;" />

<details>
<summary><h1>🦮 COMMAND LINE TRAINING</h1></summary>

We also provide the possibility to use your custom model for inference. After training the model, the model is saved in a folder whose path can be given to the inference commands as a `--custom-model` option.

To train your custom model you will need to run two commands: `create-training-data` to generate labels and PLM embeddings, and `train-model` to train the model.

<details>
<summary><h2>📁 Commands</h2></summary>

<details>
<summary><h3>1. create-training-data - Generate Training Dataset</h3></summary>

Create dataset to train the neural network. Sequences and labels are saved in a .json file, and LPLM embeddings are saved in a .pt file.

#### Usage
```bash
create-training-data [OPTIONS] CSV_FILE_PATH
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
<summary><h4>Examples</h4></summary>

**Basic usage:**
```bash
create-training-data data/training_sequences.csv pdb_folder
```

**With custom settings:**
```bash
create-training-data data/training_sequences.csv pdb_folder \
  -r results/training_data \
  --gpu 0 \
  --emb-proc-size 50 \
  --single-chain
```

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
<summary><h4>Examples</h4></summary>

**Basic training:**
```bash
train-model results/training_data/train results/training_data/val
```

**Advanced training with custom parameters:**
```bash
train-model results/training_data/train results/training_data/val \
  --lr 0.001 \
  -n 50 \
  --batch-size 32 \
  --dims 512,256 \
  --dropouts 0.2,0.1 \
  --patience 5 \
  --emb-models igT5,esm \
  --gpu 0
```

</details>

</details>

</details>

<details>
<summary><h2>🚀 Complete Training Workflow</h2></summary>

Here's a complete example of training a custom model:

```bash
# Step 1: Create training data from CSV
create-training-data data/my_sequences.csv -r my_training_results --gpu 0

# Step 2: Train the model
train-model my_training_results/train my_training_results/val \
  --lr 0.001 \
  -n 50 \
  --batch-size 32 \
  --dims 512,256 \
  --dropouts 0.2,0.1 \
  --patience 5 \
  --emb-models igT5,esm \
  --gpu 0
```

After training, your custom model will be saved in the results folder and can be used with inference commands using the `--custom-model` option.

</details>

</details>

<hr style="height:3px;border:none;background-color:#ff6b6b;" />

# ⚡ QUICK START

1. **Install**: `pip install paraplume`
2. **Single sequence**: `infer-paratope seq-to-paratope -h YOUR_HEAVY_CHAIN -l YOUR_LIGHT_CHAIN`
3. **File batch**: `infer-paratope file-to-paratope your_file.csv`

For detailed usage, expand the sections above! 👆

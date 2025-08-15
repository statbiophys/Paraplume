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
<summary><h1>💻 COMMAND LINE </h1></summary>
We provide several commands to use the model as inference with the default weights or retrain the model with a custom dataset. All commands can be run with cpu or gpu, if available (cf gpu option).

`paraplume-infer` provides two commands, one to infer the paratope from a unique sequence (`seq-to-paratope`) and another from a batch of sequences in the form of a csv file (`file-to-paratope`).
```bash
paraplume-infer COMMAND [OPTIONS][ARGS] ...
```
By default the model used is trained using the 'expanded' dataset from the [Paragraph](https://academic.oup.com/bioinformatics/article/39/1/btac732/6825310) paper, that we divided in 1000 sequences for the training set and 85 sequences for the validation and available in `./datasets/`. PDB `4FQI` was excluded from the train and validation sets as we analyze variants of this antibody in our paper using the trained model.

However we also provide the possibility to use a custom model for inference. To train your custom model you will need to run two commands: `paraplume-create-dataset` to generate labels and PLM embeddings for your desired training dataset, and `paraplume-train` to train the model.

After training the model on your custom dataset, the model is saved in a folder whose path can be given to the inference commands as a `--custom-model` option.
<details>
<summary><h2>📋 Commands</h2></summary>

<details>
<summary><h3>1. paraplume-infer seq-to-paratope</h3></summary>

Predict paratope directly from amino acid sequences provided as command line arguments.

#### Usage
```bash
paraplume-infer seq-to-paratope [OPTIONS]
```

#### Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-h, --heavy-chain` | TEXT | - | Heavy chain amino acid sequence |
| `-l, --light-chain` | TEXT | - | Light chain amino acid sequence |
| `--custom-model` | PATH | None | Path to custom trained model folder |
| `--gpu` | INT | 0 | Choose index of GPU device to use if multiple GPUs available. By default it's the first one (index 0). -1 forces cpu usage. If no GPU is available, CPU is used |
| `--large/--small` | flag | --large | Use default Paraplume which uses the 6 PLMs AbLang2,Antiberty,ESM,ProtT5,IgT5 and IgBert (--large) or the smallest version using only ESM-2 embeddings (--small) |

<details>
<summary><h4>Examples</h4></summary>

**Both chains:**
```bash
paraplume-infer seq-to-paratope \
  -h QAYLQQSGAELVKPGASVKMSCKASDYTFTNYNMHWIKQTPGQGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCASLGSSYFDYWGQGTTLTVSS \
  -l EIVLTQSPTTMAASPGEKITITCSARSSISSNYLHWYQQKPGFSPKLLIYRTSNLASGVPSRFSGSGSGTSYSLTIGTMEAEDVATYYCHQGSNLPFTFGSGTKLEIK
```

**Heavy chain only:**
```bash
paraplume-infer seq-to-paratope \
  -h QAYLQQSGAELVKPGASVKMSCKASDYTFTNYNMHWIKQTPGQGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCASLGSSYFDYWGQGTTLTVSS
```

**Light chain only:**
```bash
paraplume-infer seq-to-paratope \
  -l EIVLTQSPTTMAASPGEKITITCSARSSISSNYLHWYQQKPGFSPKLLIYRTSNLASGVPSRFSGSGSGTSYSLTIGTMEAEDVATYYCHQGSNLPFTFGSGTKLEIK
```

</details>

</details>

<details>
<summary><h3>2. paraplume-infer file-to-paratope</h3></summary>

Predict paratope from sequences stored in a CSV file.

#### Usage
```bash
paraplume-infer file-to-paratope [OPTIONS] FILE_PATH
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
| `--gpu` | INT | 0 | Choose index of GPU device to use if multiple GPUs available. By default it's the first one (index 0). -1 forces cpu usage. If no GPU is available, CPU is used |
| `--result_folder, -r` | PATH | result | Folder path where to save the results. If not passed the result is saved in the input data folder |
| `--emb-proc-size` | INT | 100 | Embedding batch size for memory management |
| `--compute-sequence-embeddings` | flag | False | Compute both paratope and classical sequence embeddings for each sequence and each of the 6 PLMs AbLang2, Antiberty, ESM, ProtT5, IgT5 and IgBert. Only possible when using the default trained_models/large |
| `--single-chain` | flag | False | Process single chain sequences |
| `--large/--small` | flag | --large | Use default Paraplume which uses the 6 PLMs AbLang2,Antiberty,ESM,ProtT5,IgT5 and IgBert (--large) or the smallest version using only ESM-2 embeddings (--small) |



<details>
<summary><h4>Examples</h4></summary>

**Paired chains:**
```bash
paraplume-infer file-to-paratope ./tutorial/paired.csv
```

**Heavy chain only:**
```bash
paraplume-infer file-to-paratope ./tutorial/heavy.csv --single-chain
```

**Light chain only:**
```bash
paraplume-infer file-to-paratope ./tutorial/light.csv --single-chain
```

Sample input files are available in the `tutorial` folder.

</details>

<details>
<summary><h4>Input</h4></summary>

Your CSV file must include these columns (any additional column is fine):

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

Creates a pickle file (e.g., `./tutorial/paratope_paired.pkl`) containing:
- `model_prediction_heavy` - Paratope predictions for heavy chains
- `model_prediction_light` - Paratope predictions for light chains

**Reading results:**
```python
import pandas as pd
predictions = pd.read_pickle("./tutorial/paratope_paired.pkl")
print(predictions.head())
```

</details>

</details>

<details>
<summary><h3>3. paraplume-create-dataset</h3></summary>

Create dataset to train the neural network. Sequences and labels are saved in a `.json` file, and LPLM embeddings are saved in a `.pt` file.

#### Usage
```bash
paraplume-create-dataset [OPTIONS] CSV_FILE_PATH PDB_FOLDER_PATH
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
| `--gpu` | INTEGER | 0 | Choose index of GPU device to use if multiple GPUs available. By default it's the first one (index 0). -1 forces cpu usage. If no GPU is available, CPU is used |
| `--single-chain` | flag | False | Generate embeddings using llms on single chain mode, which slightly increases performance |

<details>
<summary><h4>Example</h4></summary>

```bash
paraplume-create-dataset ./tutorial/custom_train_set.csv pdb_folder \
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
<summary><h3>4. paraplume-train</h3></summary>

Train the model given provided parameters and data.

#### Usage
```bash
paraplume-train [OPTIONS] TRAIN_FOLDER_PATH VAL_FOLDER_PATH
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
| `--gpu` | INTEGER | 0 | Choose index of GPU device to use if multiple GPUs available. By default it's the first one (index 0). -1 forces cpu usage. If no GPU is available, CPU is used |

<details>
<summary><h4>Example</h4></summary>

```bash
paraplume-train training_data/custom_train_set training_data/custom_val_set \
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

The two arguments (`training_data/custom_train_set` and `training_data/custom_val_set` in the example) are paths of folders created by the previous `paraplume-create-dataset` command.

</details>

<details>
<summary><h4>Output</h4></summary>

Model weights and training parameters are saved in a folder specified by the -r option (`training_results` in the example, `results` by default).

</details>

**The resulting trained model can then be used at inference by passing the output folder path as the --custom-model argument of the inference commands (see inference command lines).**


</details>



</details>

</details>

<hr style="height:3px;border:none;background-color:#ff6b6b;" />

<details>
<summary><h1>🚀 TUTORIALS </h1></summary>

<details>
<summary><h2>Command Line Tutorial</h2></summary>

If you want to use the default model with the already trained weights, just install the package and run `paraplume-infer file-to-paratope ./tutorial/paired.csv` and the result will be available as `paratope_paired.pkl` in the same `tutorial` folder.

If you want to train and use your custom model via command line, follow the 4 steps below.

#### Step 0: Set up
- Clone repository
- Make sure you are in `Paraplume`.
- Install the package in your favorite virtual environment with `pip install paraplume`
- Download PDB files from [SabDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/about#formats) using IMGT format and save them in `./all_structures/imgt`.

#### Step 1: Create training and validation datasets from CSVs
```bash
paraplume-create-dataset ./tutorial/custom_train_set.csv ./all_structures/imgt -r custom_folder
```
The folder `custom_folder` will be created. Inside this folder the folder `custom_train_set` is created in which there are two files, `dict.json` for the sequences and labels, and `embeddings.pt` for the PLM embeddings.
Repeat for the validation set (used for early stopping):
```bash
paraplume-create-dataset ./tutorial/custom_val_set.csv ./all_structures/imgt -r custom_folder
```

#### Step 2: Train the model
```bash
paraplume-train ./custom_folder/custom_train_set ./custom_folder/custom_val_set \
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

#### Step 3: Use the trained custom model for inference
After training, your custom model will be saved in the results folder and can be used with inference commands using the `--custom-model` option.

```bash
paraplume-infer file-to-paratope ./tutorial/paired.csv --custom-model ./custom_folder
```

And the result is available as `paratope_paired.pkl` in the `tutorial` folder !!

</details>

<details>
<summary><h2>Python Tutorial</h2></summary>

A comprehensive Python tutorial for default inference usage (using the already trained weights) with examples is available in the `tutorial` folder.

If you want to use to train and use your custom model, follow the command line tutorial, or use the code available in `paraplume/create_dataset.py` and `paraplume/train.py` (function main in both files). Don't hesitate to contact me if you need help **gabrielathenes@gmail.com**.

</details>

</details>

<hr style="height:3px;border:none;background-color:#ff6b6b;" />

# ⚡ QUICK START

1. **Install**: `pip install paraplume`
2. **Single sequence**: `paraplume-infer seq-to-paratope -h YOUR_HEAVY_CHAIN -l YOUR_LIGHT_CHAIN`
3. **File batch**: `paraplume-infer file-to-paratope your_file.csv`

For detailed usage, expand the sections above! 👆

# 📧 Contact

Any issues or questions should be addressed to us at **gabrielathenes@gmail.com**.

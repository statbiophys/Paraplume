<h1 align="center">
  <img src="doc/logo.png" width="500">
</h1>

**Paraplume** is a sequence-based paratope prediction method. It predicts which amino acids in an antibody sequence are likely to interact with an antigen during binding. Concretely, given an amino acid sequence, the model returns a **probability for each residue** indicating the likelihood of antigen interaction.
Go check out our [paper](https://arxiv.org/abs/2510.05626) to see how Paraplume can be used to study antibody evolution/function !

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

However we also provide the possibility to use a custom model for inference. To train your custom model you will need to run three commands: `paraplume-build-dictionary` to generate labels, `paraplume-create-embeddings` to create PLM embeddings for your desired training dataset, and `paraplume-train` to train the model.

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
| `--result-folder, -r` | PATH | None | Folder path where to save the results. If not passed the result is saved in the input data folder |
| `--emb-proc-size` | INT | 100 | Embedding batch size for memory management |
| `--compute-sequence-embeddings` | flag | False | Compute both paratope and classical sequence embeddings for each sequence and each of the 6 PLMs AbLang2, Antiberty, ESM, ProtT5, IgT5 and IgBert. Only possible when using the default trained_models/large |
| `--single-chain` | flag | False | Process single chain sequences |
| `--large/--small` | flag | --large | Use default Paraplume which uses the 6 PLMs AbLang2,Antiberty,ESM,ProtT5,IgT5 and IgBert (--large) or the smallest version using only ESM-2 embeddings (--small) |
│ `--compute-shap` | flag | False | Compute SHAP importance analysis and generate visualizations. A folder 'shap_results' will be created with a plot inside for each sequence.|



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
<summary><h3>3. paraplume-build-dictionary</h3></summary>

Create dataset to train the neural network. Sequences and labels are saved in a .json file, and LPLM embeddings are saved in a .pt file.

#### Usage
```bash
paraplume-build-dictionary [OPTIONS] CSV_FILE_PATH PDB_FOLDER_PATH
```

#### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| CSV_FILE_PATH | PATH | ✓ | Path of csv file to use for pdb list |
| PDB_FOLDER_PATH | PATH | ✓ | Pdb path for ground truth labeling |

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| --result-folder, -r | PATH | result | Where to save results |
| --help | flag | - | Show this message and exit |

<details>
<summary><h4>Example</h4></summary>

```bash
paraplume-build-dictionary ./tutorial/custom_train_set.csv pdb_folder -r training_data
```

</details>

<details>
<summary><h4>Input</h4></summary>

custom_train_set.csv contains information about the PDB files used for training and has the following format:

| pdb | Lchain | Hchain | antigen_chain |
|------|--------|--------|---------------|
| 1ahw | D | E | F |
| 1bj1 | L | H | W |
| 1ce1 | L | H | P |

**Column descriptions:**
- **pdb**: PDB code of the antibody-antigen complex (should be available in pdb_folder as `pdb_folder/pdb_code.pdb`)
- **Lchain**: Light chain identifier used to label the paratope
- **Hchain**: Heavy chain identifier used to label the paratope
- **antigen_chain**: Antigen chain identifier used to label the paratope

</details>

<details>
<summary><h4>Output</h4></summary>

Creates a folder with the same name as the CSV file (e.g., `custom_train_set`) inside the result folder (`training_data`), which contains **dict.json**: Sequences and labels for each structure

</details>

</details>

<details>
<summary><h3>4. paraplume-create-embeddings</h3></summary>

Generate PLM embeddings from a dictionary file created by `paraplume-build-dictionary` and saves them
in the folder of the dictionary.

#### Usage
```bash
paraplume-create-embeddings [OPTIONS] DICT_PATH
```

#### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| DICT_PATH | PATH | ✓ | Path to the folder containing dict.json |

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| --emb-proc-size | INTEGER | 100 | Chunk size for creating embeddings to avoid memory explosion. Optimal value depends on your computer |
| --gpu | INTEGER | 0 | Choose index of GPU device to use if multiple GPUs available. By default it's the first one (index 0). -1 forces cpu usage. If no GPU is available, CPU is used |
| --single-chain | flag | False | Generate embeddings using LLMs on single chain mode, which slightly increases performance |
| --help | flag | - | Show this message and exit |

<details>
<summary><h4>Example</h4></summary>

```bash
paraplume-create-embeddings ./training_data/custom_train_set/dict.json \
  --gpu 0 \
  --emb-proc-size 50 \
  --single-chain
```

</details>

<details>
<summary><h4>Input</h4></summary>

Path of **dict.json**: Dictionary file created by `paraplume-build-dictionary` with sequences and labels

</details>

<details>
<summary><h4>Output</h4></summary>

Creates multiple embedding files in the same folder as dict.json:
- **ablang2_embeddings.pt**: AbLang2 model embeddings
- **igbert_embeddings.pt**: IgBERT model embeddings
- **igT5_embeddings.pt**: IgT5 model embeddings
- **esm_embeddings.pt**: ESM model embeddings
- **antiberty_embeddings.pt**: AntiBERTy model embeddings
- **prot-t5_embeddings.pt**: ProtT5 model embeddings

</details>

</details>

<details>
<summary><h3>5. paraplume-train</h3></summary>

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
| `--result-folder, -r` | PATH | result | Where to save results |
| `--pos-weight` | FLOAT | 1 | Weight to give to positive labels |
| `--batch-size, -bs` | INTEGER | 10 | Batch size |
| `--mask-prob` | FLOAT | 0 | Probability with which to mask each embedding coefficient |
| `--dropouts` | TEXT | 0 | Dropout probabilities for each hidden layer, separated by commas. Example '0.3,0.3' |
| `--dims` | TEXT | 1000 | Dimensions of hidden layers. Separated by commas. Example '100,100' |
| `--override` | flag | False | Override results |
| `--seed` | INTEGER | 0 | Seed to use for training |
| `--l2-pen` | FLOAT | 0 | L2 penalty to use for the model weights |
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

The two arguments (`training_data/custom_train_set` and `training_data/custom_val_set` in the example) are paths of folders created by the previous `paraplume-build-dictionary` command.

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
paraplume-build-dictionary ./tutorial/custom_train_set.csv ./all_structures/imgt -r custom_folder
```
followed by
```bash
paraplume-create-embeddings ./custom_folder/custom_train_set/dict.json \
  --gpu 0 \
  --emb-proc-size 50
```

The folder `custom_folder` will be created. Inside this folder the folder `custom_train_set` is created in which there are two files, `dict.json` for the sequences and labels, and emebddings for each of the 6 PLM.
Repeat for the validation set (used for early stopping):
```bash
paraplume-build-dictionary ./tutorial/custom_val_set.csv ./all_structures/imgt -r custom_folder
```
followed by
```bash
paraplume-create-embeddings ./custom_folder/custom_val_set/dict.json \
  --gpu 0 \
  --emb-proc-size 50
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

If you want to use to train and use your custom model, follow the command line tutorial. Don't hesitate to contact me if you need help **gabrielathenes@gmail.com**.

</details>
</details>
<hr style="height:3px;border:none;background-color:#ff6b6b;" />
<details>
<summary><h1>📊 BENCHMARK</h1></summary>

The benchmark was conducted using **Paraplume v1.0.0**. The final model configuration used a learning rate of
\(1 \times 10^{-5}\), a batch size of 16 sequences, and the ADAM optimizer with an L2 regularization weight of
\(1 \times 10^{-5}\). The MLP architecture consisted of three hidden layers with widths of 2000, 1000, and 500.
A summary of the explored hyperparameter ranges and the selected values is provided in **Table S4** of the
[paper](https://arxiv.org/abs/2510.05626).

All experiments were performed on a workstation equipped with two NVIDIA RTX 5000 Ada
GPUs (32 GB VRAM each). Models were trained and evaluated using random seeds 1 through 16, and all reported
results correspond to averages across these seeds.

All scripts and generated data are publicly available on the
[Zenodo repository](https://zenodo.org/records/17021232).
</details>


<hr style="height:3px;border:none;background-color:#ff6b6b;" />

<details>
<summary><h1>🔍 INTERPRETABILITY</h1></summary>

Predictions and PLM importance over residues can be visualized with the `--compute-shap`  option of `paraplume-infer file-to-paratope`.

<h1 align="center">
  <img src="doc/intepretability.png" width="500">
</h1>

</details>


<hr style="height:3px;border:none;background-color:#ff6b6b;" />


# ⚡ QUICK START

1. **Install**: `pip install paraplume`
2. **Single sequence**: `paraplume-infer seq-to-paratope -h YOUR_HEAVY_CHAIN -l YOUR_LIGHT_CHAIN`
3. **File batch**: `paraplume-infer file-to-paratope your_file.csv`

For detailed usage, expand the sections above! 👆

# 🛠️ Troubleshooting & Notes

- During the **first** inference, Paraplume will automatically download PLM weights inside your virtual environment. This step may take **10–15 minutes**, depending on connection and hardware.
- This download only happens **once**. Future runs will start right away.
- If the full model is too heavy for your system, try the **light version** by adding `--small`, which uses only ESM.

<details>
<summary><h2>Common Issues</h2></summary>

#### AbLang2 Download Error

If you encounter the following error:
```
CalledProcessError: Command '['tar', '-zxvf', 'YOURFOLDER/tmp.tar.gz',
'-C', 'YOURFOLDER']' returned non-zero exit status 2.
```

This occurs because AbLang2 failed to download its model weights from the Zenodo server.

**Fix:** Download the weights manually:
```bash
TARGET=YOURFOLDER
mkdir -p "$TARGET"
curl -L "https://zenodo.org/records/10185169/files/ablang2-weights.tar.gz" | tar -xz -C "$TARGET"
```

Replace `YOURFOLDER` with the actual path shown in your error message. After running these commands, Paraplume should work correctly.

</details>

# 📧 Contact

Any issues or questions should be addressed to us at **gabrielathenes@gmail.com**.

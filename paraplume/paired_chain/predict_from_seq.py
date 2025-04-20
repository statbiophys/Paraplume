"""Create dictionary of sequence/labels and corresponding LLM embeddings."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from create_dataset import create_emebddings_from_seq
from torch.nn import Dropout, Linear, ReLU, Sequential, Sigmoid

app = typer.Typer(add_completion=False)
warnings.filterwarnings("ignore")
from paraplume.utils import get_logger
from paraplume.utils_paired import get_embedding_paired

log = get_logger()

# pylint: disable=E1101

@app.command()
def predict_from_df(
    file_path:Path=typer.Argument(
        ...,
        help="Path of the file.",
        show_default=False,
    ),
    result_folder: Path = typer.Argument(
        ...,
        help="Path of the folder where the model is saved.",
        show_default=False,
    ),
    name:str=typer.Option(
        "",
        "--name",
        help="Extension to add to the file.",
    ),
    save_embeddings:bool=typer.Option(
        False,
        "--save-embeddings",
        help="Whether to save the embeddings or not.\
            Not advised for large files."
        ),
    gpu:int=typer.Option(
        1,
        "--gpu",
        help="Which GPU to use."
    )) -> None:
    """Predict paratope from sequence."""
    summary_dict_path = result_folder / Path("summary_dict.json")
    log.info("Loading training summary dictionary.", path=summary_dict_path.as_posix())
    with open(summary_dict_path, encoding="utf-8") as f:
        summary_dict = json.load(f)

    layers=[]
    input_size = int(summary_dict["input_size"])
    dims = [int(each) for each in summary_dict["dims"].split(",")]
    dropouts=[0]*len(dims)
    for i, _ in enumerate(dims):
        if i==0:
            layers.append(Linear(input_size, dims[i]))
            layers.append(Dropout(dropouts[i]))
            layers.append(ReLU())
        else :
            layers.append(Linear(dims[i-1], dims[i]))
            layers.append(Dropout(dropouts[i]))
            layers.append(ReLU())
    model=Sequential(Sequential(*layers), Sequential(Linear(dims[-1],1),Sigmoid()))
    model_path = result_folder / Path("checkpoint.pt")
    log.info("Loading model.", path=model_path.as_posix())
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    embedding_models = summary_dict["embedding_models"]
    if embedding_models == "all":
        embedding_models = "ablang2,igbert,igT5,esm,antiberty,prot-t5"
    embedding_models_list = embedding_models.split(",")

    df = pd.read_csv(file_path)
    sequences_heavy = df["sequence_heavy"].tolist()
    sequences_light = df["sequence_light"].tolist()
    embeddings = create_emebddings_from_seq(sequences_heavy,sequences_light,emb_proc_size=50)
    if save_embeddings:
        torch.save(embeddings, f"{str(file_path.parents[0])}/embeddings.pt")

    heavy_outputs=[]
    light_outputs=[]
    embeddings_classical = []
    embeddings_paratope=[]
    for i, embedding in enumerate(embeddings):
        heavy, light = len(sequences_heavy[i]), len(sequences_light[i])
        emb = get_embedding_paired(
            embedding=embedding,
            embedding_models=embedding_models_list,
            heavy=heavy,
            light=light,
        )
        emb=emb.to(device)
        output=np.round(model(emb).cpu().detach().numpy().flatten().astype(np.float64), 12).tolist()
        heavy_outputs.append(output[:heavy])
        light_outputs.append(output[heavy:])
        emb_classical = emb.sum(0)/emb.shape[0]
        emb_classical=np.round(emb_classical.cpu().detach().numpy().flatten().astype(np.float64), 12).tolist()
        embeddings_classical.append(emb_classical)
        emb_paratope = np.zeros(int(emb.shape[-1]))
        normalized_output = output/np.sum(output)
        for prob, embed in zip(normalized_output,emb):
            emb_paratope+=prob*embed.cpu().detach().numpy()
        emb_paratope = np.round(emb_paratope.flatten().astype(np.float64), 12).tolist()
        embeddings_paratope.append(emb_paratope)

    df["embeddings_paratope"]=embeddings_paratope
    df["embeddings_classical"]=embeddings_classical
    df["model_prediction_heavy"]=heavy_outputs
    df["model_prediction_light"]=light_outputs
    result_path = file_path.parents[0] / Path(f"{name}paratope_"+file_path.name)
    df.to_csv(result_path)
    return output

@app.command()
def predict_from_sequence(
    sequence_heavy: str = typer.Argument(
        ...,
        help="Path of csv file to use for pdb list.",
        show_default=False,
    ),
    sequence_light: str = typer.Argument(
        ...,
        help="Path of csv file to use for pdb list.",
        show_default=False,
    ),
    result_folder: Path = typer.Argument(
        ...,
        help="Path of the result folder.",
        show_default=False,
    ),) -> None:
    """Predict paratope from sequence."""
    summary_dict_path = result_folder / Path("summary_dict.json")
    log.info("Loading training summary dictionary.", path=summary_dict_path.as_posix())
    with open(summary_dict_path, encoding="utf-8") as f:
        summary_dict = json.load(f)

    layers=[]
    input_size = int(summary_dict["input_size"])
    dims = [int(each) for each in summary_dict["dims"].split(",")]
    dropouts=[0]*len(dims)
    for i, _ in enumerate(dims):
        if i==0:
            layers.append(Linear(input_size, dims[i]))
            layers.append(Dropout(dropouts[i]))
            layers.append(ReLU())
        else :
            layers.append(Linear(dims[i-1], dims[i]))
            layers.append(Dropout(dropouts[i]))
            layers.append(ReLU())
    model=Sequential(Sequential(*layers), Sequential(Linear(dims[-1],1),Sigmoid()))
    model_path = result_folder / Path("checkpoint.pt")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    embedding = create_emebddings_from_seq([sequence_heavy],[sequence_light],emb_proc_size=1)
    heavy, light = len(sequence_heavy), len(sequence_light)
    embedding_models = summary_dict["embedding_models"]
    if embedding_models == "all":
        embedding_models = "ablang2,igbert,igT5,esm,antiberty,prot-t5"
    embedding_models_list = embedding_models.split(",")
    emb = get_embedding_paired(
        embedding=embedding[0],
        embedding_models=embedding_models_list,
        heavy=heavy,
        light=light,
    )
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    emb=emb.to(device)
    output = model(emb)
    return output

if __name__ == "__main__":
    app()

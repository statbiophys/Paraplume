import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import typer
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from graph_torch_dataset import create_graph_dataloader
from models import EarlyStopping, EGNN_Model
from utils import get_dim, save_plot

app = typer.Typer(add_completion=False)

import warnings

warnings.filterwarnings("ignore")

def evaluate(model, test_loader,infer_edges:bool=False):
    """_summary_

    Args:
        model (_type_): _description_
        test_loader (_type_): _description_
        embedding_models (_type_): _description_
        infer_edges (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    all_outputs = []
    all_targets = []
    model=model.to(device)
    with torch.no_grad():
        model.eval()
        for (feats, coors, edges), labels, _,_,_ in tqdm(test_loader):
            feats=feats.to(device)
            coors=coors.to(device)
            edges=edges.to(device)
            labels=labels.to(device)
            if infer_edges:
                pred = model(feats, coors, edges=None)
            else:
                pred = model(feats, coors, edges)
            labels = labels.squeeze()
            out = torch.sigmoid(pred)
            out = out.squeeze()

            out = out.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            all_outputs.extend(out)
            all_targets.extend(labels)


    # Converting lists to numpy arrays for AUC and ROC calculation
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)

    # Calculate the AUC score
    auc = roc_auc_score(all_targets, all_outputs)
    ap = average_precision_score(all_targets, all_outputs)
    return auc, ap


@app.command()
def main(
    result_folder: Path = typer.Argument(
        ...,
        help="Path of model.",
        show_default=False,
    ),
    test_folder_path: Path = typer.Argument(
        ...,
        help="Path of testfolder.",
        show_default=False,
    ),
    test_csv_path: Path = typer.Argument(
        ...,
        help="Path of test csv.",
        show_default=False),
    pdb_folder_path_test: Path = typer.Option(
        "/home/gathenes/paragraph_benchmark/abb3_pdbs_renumbered",
        "--pdb-folder-path-test",
        help="Pdb folder path.",
    ),
    override: bool = typer.Option(
        False, "--override", help="Override results. Defaults to False"
    ),
) -> None:
    if (result_folder / Path("graph_results_dict.json")).exists() and not override:
        print("Not overriding results.")
        return
    args_dict = {
        "test_folder_path": str(test_folder_path),
        "pdb_folder_path_test": str(pdb_folder_path_test),
        "result_folder": str(result_folder),
        "test_csv_path":str(test_csv_path),
    }
    model_path = result_folder / Path("graph_checkpoint.pt")
    with open(result_folder/Path("graph_summary_dict.json"), encoding="utf-8") as f:
        summary_dict=json.load(f)
    with open(test_folder_path / Path("dict.json"), encoding="utf-8") as f:
        dict_test = json.load(f)
    test_embeddings = torch.load(
        test_folder_path / Path("embeddings.pt"), weights_only=True
    )
    test_csv = pd.read_csv(test_csv_path)
    embedding_models = summary_dict["embedding_models"]
    infer_edges = summary_dict["infer-edges"]
    if infer_edges:
        edge_dim = 0
    else:
        edge_dim = 1
    dropout=summary_dict["dropout"]
    if embedding_models=='all':
        embedding_models="ablang2,igbert,igT5,esm,antiberty,prot-t5"
    embedding_models_list = embedding_models.split(",")
    graph_distance=summary_dict["graph_distance"]
    region=summary_dict["region"]
    test_loader = create_graph_dataloader(
        csv=test_csv,
        dataset_dict=dict_test,
        embeddings=test_embeddings,
        shuffle=False,
        pdb_folder_path=pdb_folder_path_test,
        embedding_models=embedding_models_list,
        region=region,
        graph_distance=graph_distance,
    )
    if embedding_models == "one-hot":
        feature_dim = 22
    else:
        feature_dim=get_dim(embedding_models_list)

    num_graph_layers = summary_dict["num_graph_layers"]
    linear_layers_dims = summary_dict["linear_layers_dims"]
    graph_hidden_layer_output_dims = [feature_dim] * num_graph_layers
    linear_hidden_layer_output_dims = [int(x) for x in linear_layers_dims.split(",")]
    model = EGNN_Model(
        num_feats=feature_dim,
        graph_hidden_layer_output_dims=graph_hidden_layer_output_dims,
        linear_hidden_layer_output_dims=linear_hidden_layer_output_dims,
        edge_dim=edge_dim,
        dropout=dropout,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))

    auc, ap = evaluate(
            model=model,
            test_loader=test_loader,
            infer_edges=infer_edges,
        )
    args_dict["ap"]=ap
    args_dict["auc"]=auc

    with open(result_folder / Path("graph_results_dict.json"), "w", encoding="utf-8") as json_file:
        json.dump(args_dict, json_file, indent=4)


if __name__ == "__main__":
    app()

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

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    n_epochs=3,
    model_save_path=Path("./checkpoint.pt"),
    criterion=nn.BCELoss(),
    infer_edges=False,
    batch_size=16,
    patience:int=100,
    mask_prob=0,
):
    # Training loop
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model=model.to(device)
    train_loss_list = []
    val_loss_list = []
    auc_list = []
    ap_list = []

    early_stopping = EarlyStopping(patience=patience, path=model_save_path, best_score=0)
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for j, ((feats, coors, edges), labels, _,_,_) in enumerate(tqdm(train_loader)):
            drop_mask = torch.rand(feats.size(), device=feats.device) >= mask_prob
            feats=feats * drop_mask.float()
            feats=feats.to(device)
            coors=coors.to(device)
            edges=edges.to(device)
            labels=labels.to(device)
            if (j + 1) % batch_size == 0 or (j + 1) == len(train_loader):
                optimizer.zero_grad()
            if infer_edges:
                pred = pred = model(feats, coors, edges=None)
            else:
                pred = model(feats, coors, edges)
            labels = labels.squeeze()
            pred = pred.squeeze()
            loss = criterion(pred, labels)
            loss = loss / batch_size
            loss.backward()
            if (j + 1) % batch_size == 0 or (j + 1) == len(train_loader):
                optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {train_loss}")

        val_loss = 0
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            model.eval()
            for (feats, coors, edges), labels, _,_,_ in tqdm(val_loader):
                feats=feats.to(device)
                coors=coors.to(device)
                edges=edges.to(device)
                labels=labels.to(device)
                if infer_edges:
                    pred = model(feats, coors, edges=None)
                else:
                    pred = model(feats, coors, edges)
                labels = labels.squeeze()
                loss = criterion(pred.squeeze(), labels)
                out = torch.sigmoid(pred)
                out = out.squeeze()
                val_loss += loss.item()

                out = out.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                all_outputs.extend(out)
                all_targets.extend(labels)

            val_loss /= len(val_loader)
            val_loss_list.append(val_loss)

        # Converting lists to numpy arrays for AUC and ROC calculation
        all_outputs = np.array(all_outputs)
        all_targets = np.array(all_targets)

        # Calculate the AUC score
        auc = roc_auc_score(all_targets, all_outputs)
        auc_list.append(auc)

        ap = average_precision_score(all_targets, all_outputs)
        ap_list.append(ap)

        # Calculate the Average Precision score
        early_stopping(-ap, model)
        if early_stopping.early_stop:
            print(f"Early stopping, last epoch = {epoch}")
            break
    return train_loss_list, val_loss_list, auc_list, ap_list


@app.command()
def main(
    train_folder_path: Path = typer.Argument(
        ...,
        help="Path of trainfolder.",
        show_default=False,
    ),
    val_folder_path: Path = typer.Argument(
        ...,
        help="Path of valfolder.",
        show_default=False,
    ),
    train_csv_path:Path=typer.Argument(
        ...,
        help="Path of train csv.",
        show_default=False,
    ),
    val_csv_path:Path=typer.Argument(
        ...,
        help="Path of val csv.",
        show_default=False,
    ),
    learning_rate: float = typer.Option(
        0.001, "--lr", help="Learning rate to use for training."
    ),
    n_epochs: int = typer.Option(
        1, "--n_epochs", "-n", help="Number of n_epochs to use for training."
    ),
    result_folder: Path = typer.Option(
        Path("./result/"), "--result_folder", "-r", help="Where to save results."
    ),
    positive_weight: float = typer.Option(
        3, "--pos-weight", help="Weight to give to positive labels."
    ),
    batch_size: int = typer.Option(
        16, "--batch-size", "-bs", help="Batch size. Defaults to 10."
    ),
    override: bool = typer.Option(
        False, "--override", help="Override results. Defaults to False"
    ),
    alpha: str = typer.Option(
        4.5, "--alpha", help="Alpha distance to use for labels. Default to 4.5."
    ),
    seed: int = typer.Option(0, "--seed", help="Seed to use for training."),
    pdb_folder_path_val: Path = typer.Option(
        "/home/gathenes/paragraph_benchmark/abb3_pdbs_renumbered",
        "--pdb-folder-path-val",
        help="Pdb folder path.",
    ),
    pdb_folder_path_train: Path = typer.Option(
        "/home/gathenes/all_structures/imgt_renumbered_expanded/",
        "--pdb-folder-path-train",
        help="Pdb folder path.",
    ),
    embedding_models:str=typer.Option(
        "one-hot", "--emb-models", help="Embedding models to use, separated by commas. \
            Models should be in 'ablang2','igbert','igT5','esm','antiberty',prot-t5','all'. \
            All means all of the preceding models. One-hot means the traditional one hot encoding" \
            "of amino acid. Default to one-hot"
    ),
    infer_edges: bool = typer.Option(
        False, "--infer-edges", help="Infer edges instead of using sparse graph."
    ),
    dropout:float=typer.Option(
        0,"--dropout", help="Dropout for EGNN. Defaults to 0."
    ),
    num_graph_layers:int=typer.Option(
        6, "--num-graph-layers", help="Graph layers dimensions for EGNN. \
        Defaults to 6."),
    linear_layers_dims:str=typer.Option(
        "10,10", "--linear-layers-dims", help="Linear layers dimensions for EGNN. \
            Defaults to 10,10."),
    mask_prob:float=typer.Option(
        0, "--mask-prob", help="Probability with which to mask each embedding coefficient. \
            Defaults to 0"
    ),
    patience:int=typer.Option(
        100, "--patience", help="Patience to use for early stopping. 0 means no early stopping. \
        Defaults to 0."
    ),
    l2_pen:float=typer.Option(
        0, "--l2-pen", help="L2 penalty to use for the model weights."
    ),
    graph_distance:float=typer.Option(
        10, "--graph-distance", help="Distance to use to compute graph"
    ),
    region:str=typer.Option(
        "cdrs","--region", help="Region to use for training. Defaults to cdrs.")
) -> None:
    if seed>0:
        torch.manual_seed(seed)
    if (result_folder / Path("graph_summary_dict.json")).exists() and not override:
        print("Not overriding results.")
        return
    elif (result_folder / Path("summary_dict.json")).exists():
        print("Overriding results.")
    result_folder.mkdir(exist_ok=True, parents=True)
    args_dict = {
        "train_folder_path": str(train_folder_path),
        "val_folder_path": str(val_folder_path),
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "result_folder": str(result_folder),
        "positive_weight": positive_weight,
        "batch_size": batch_size,
        "override": override,
        "alpha": alpha,
        "seed": seed,
        "embedding_models": embedding_models,
        "infer-edges": infer_edges,
        "pdb_folder_path_val": str(pdb_folder_path_val),
        "pdb_folder_path_train": str(pdb_folder_path_train),
        "dropout":dropout,
        "patience":patience,
        "l2_pen": l2_pen,
        "num_graph_layers": num_graph_layers,
        "linear_layers_dims": linear_layers_dims,
        "mask_prob": mask_prob,
        "graph_distance": graph_distance,
        "region": region,
    }
    if embedding_models=='all':
        embedding_models="ablang2,igbert,igT5,esm,antiberty,prot-t5"
    embedding_models_list = embedding_models.split(",")
    if seed > 0:
        torch.manual_seed(seed)
    with open(train_folder_path / Path("dict.json"), encoding="utf-8") as f:
        dict_train = json.load(f)
    train_embeddings = torch.load(
        train_folder_path / Path("embeddings.pt"), weights_only=True
    )

    with open(val_folder_path / Path("dict.json"), encoding="utf-8") as f:
        dict_val = json.load(f)
    val_embeddings = torch.load(
        val_folder_path / Path("embeddings.pt"), weights_only=True
    )

    if embedding_models == "one-hot":
        feature_dim = 22
    else:
        feature_dim=get_dim(embedding_models_list)

    if infer_edges:
        edge_dim = 0
    else:
        edge_dim = 1

    train_csv = pd.read_csv(train_csv_path)
    val_csv = pd.read_csv(val_csv_path)
    train_loader = create_graph_dataloader(
        csv=train_csv,
        dataset_dict=dict_train,
        embeddings=train_embeddings,
        shuffle=True,
        pdb_folder_path=pdb_folder_path_train,
        embedding_models=embedding_models_list,
        graph_distance=graph_distance,
        region=region,
    )
    val_loader = create_graph_dataloader(
        csv=val_csv,
        dataset_dict=dict_val,
        embeddings=val_embeddings,
        pdb_folder_path=pdb_folder_path_val,
        embedding_models=embedding_models_list,
        graph_distance=graph_distance,
        region=region,
    )
    pos_weight = torch.tensor(
        [positive_weight], device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    )  # All weights are equal to 1
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    graph_hidden_layer_output_dims = [feature_dim] * num_graph_layers
    linear_hidden_layer_output_dims = [int(x) for x in linear_layers_dims.split(",")]
    model = EGNN_Model(
        num_feats=feature_dim,
        graph_hidden_layer_output_dims=graph_hidden_layer_output_dims,
        linear_hidden_layer_output_dims=linear_hidden_layer_output_dims,
        edge_dim=edge_dim,
        dropout=dropout,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_pen)
    model_save_path = result_folder / Path("graph_checkpoint.pt")
    train_loss_list, val_loss_list, auc_list, ap_list = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=n_epochs,
        criterion=criterion,
        model_save_path=model_save_path,
        infer_edges=infer_edges,
        batch_size=batch_size,
        patience=patience,
        mask_prob=mask_prob,
    )
    save_plot_path = result_folder / Path("summary_plot.png")
    save_plot(
        train_loss_list=train_loss_list,
        val_loss_list=val_loss_list,
        auc_list=auc_list,
        ap_list=ap_list,
        save_plot_path=save_plot_path,
    )
    best_epoch = np.argmax(ap_list)
    args_dict["best_epoch"] = str(best_epoch + 1)
    args_dict["best_ap"] = str(ap_list[best_epoch])
    with open(result_folder / Path("graph_summary_dict.json"), "w", encoding="utf-8") as json_file:
        json.dump(args_dict, json_file, indent=4)


if __name__ == "__main__":
    app()

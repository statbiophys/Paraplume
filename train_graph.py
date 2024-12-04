import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import typer
from models import EarlyStopping, EGNN_Model
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from torch_dataset import create_graph_dataloader
from tqdm import tqdm
from utils import save_plot

app = typer.Typer(add_completion=False)



def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    n_epochs=3,
    model_save_path=Path("./checkpoint.pt"),
    criterion=nn.BCELoss(),
    infer_edges=False,
):
    # Training loop
    device = torch.device("cpu")
    train_loss_list = []
    val_loss_list = []
    auc_list = []
    ap_list = []

    early_stopping = EarlyStopping(patience=100, path=model_save_path, best_score=0)
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for j, ((feats, coors, edges), labels, _,_,_) in enumerate(tqdm(train_loader)):

            if (j + 1) % 16 == 0 or (j + 1) == len(train_loader):
                optimizer.zero_grad()
            if infer_edges:
                pred = pred = model(feats, coors, edges=None)
            else:
                pred = model(feats, coors, edges)
            labels = labels.squeeze()
            pred = pred.squeeze()
            loss = criterion(pred, labels)
            loss = loss / 16
            loss.backward()
            if (j + 1) % 16 == 0 or (j + 1) == len(train_loader):
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
        1, "--pos-weight", help="Weight to give to positive labels."
    ),
    batch_size: int = typer.Option(
        1, "--batch-size", "-bs", help="Batch size. Defaults to 10."
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
    features: str = typer.Option(
        "one-hot", "--features", help="Features to use for the nodes."
    ),
    infer_edges: bool = typer.Option(
        False, "--infer-edges", help="Infer edges instead of using sparse graph."
    ),
    dropout:float=typer.Option(
        0,"--dropout", help="Dropout for EGNN. Defaults to 0."
    ),
) -> None:
    if seed>0:
        torch.manual_seed(seed)
    if (result_folder / Path("summary_dict.json")).exists() and not override:
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
        "features": features,
        "infer-edges": infer_edges,
        "pdb_folder_path_val": str(pdb_folder_path_val),
        "dropout":dropout,
    }
    if seed > 0:
        torch.manual_seed(seed)
    with open(train_folder_path / Path("dict.json"), encoding="utf-8") as f:
        dict_train = json.load(f)
    train_embeddings = torch.load(
        train_folder_path / Path("embeddings.pt"), weights_only=True
    )
    train_csv = pd.read_csv(
        "/home/gathenes/paragraph_benchmark/expanded_dataset/train_set.csv"
    )

    with open(val_folder_path / Path("dict.json"), encoding="utf-8") as f:
        dict_val = json.load(f)
    val_embeddings = torch.load(
        val_folder_path / Path("embeddings.pt"), weights_only=True
    )
    val_csv = pd.read_csv(
        "/home/gathenes/paragraph_benchmark/expanded_dataset/val_set.csv"
    )

    if features == "ablang":
        feature_dim = 480
    elif features == "T5":
        feature_dim = 1024
    elif features == "all-llm":
        feature_dim = 2528
    else:
        feature_dim = 22

    if infer_edges:
        edge_dim = 0
    else:
        edge_dim = 1

    train_loader = create_graph_dataloader(
        csv=train_csv,
        dataset_dict=dict_train,
        embeddings=train_embeddings,
        batch_size=batch_size,
        shuffle=True,
        pdb_folder_path=pdb_folder_path_train,
        features=features,
    )
    val_loader = create_graph_dataloader(
        csv=val_csv,
        dataset_dict=dict_val,
        embeddings=val_embeddings,
        batch_size=batch_size,
        pdb_folder_path=pdb_folder_path_val,
        features=features,
    )
    pos_weight = torch.tensor(
        [positive_weight], device=torch.device("cpu")
    )  # All weights are equal to 1
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = EGNN_Model(
        num_feats=feature_dim,
        graph_hidden_layer_output_dims=[feature_dim] * 6,
        linear_hidden_layer_output_dims=[10] * 2,
        edge_dim=edge_dim,
        dropout=dropout,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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

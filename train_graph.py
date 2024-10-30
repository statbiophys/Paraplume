import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import typer
from models import EarlyStopping, TransformerGNN
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torch_dataset import AminoAcidGraphEGNN
from tqdm import tqdm
from utils import save_plot

app = typer.Typer(add_completion=False)

def create_dataloader(dataset_dict:Dict,residue_embeddings:torch.Tensor, csv, batch_size=1, shuffle:bool=False, alpha:str="4.5")->torch.utils.data.dataloader.DataLoader:
    """Take dataset_dict and embeddings and return dataloader.

    Args:
        dataset_dict (Dict): _description_
        residue_embeddings (torch.Tensor): _description_
        batch_size (int, optional): _description_. Defaults to 10.

    Returns:
        torch.utils.data.dataloader.DataLoader: Dataloader to use for training.
    """
    dataset = AminoAcidGraphEGNN(dataset_dict=dataset_dict, residue_embeddings=residue_embeddings,csv=csv, alpha=alpha)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset_loader


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    n_epochs=3,
    model_save_path=Path("/home/gathenes/all_structures/data_high_qual/checkpoint.pt"),
    criterion=nn.BCELoss(),
):
    # Training loop
    device = torch.device("cpu")
    train_loss_list = []
    val_loss_list = []
    auc_list = []
    ap_list = []

    early_stopping = EarlyStopping(patience=4, path=model_save_path, best_score=0)
    model.train()
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for (feats, coors, edges),labels in tqdm(train_loader):
            optimizer.zero_grad()
            pred = model(feats, coors, edges)
            labels=labels.squeeze()
            pred=pred.squeeze()
            loss=criterion(pred,labels)
            loss.backward()
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
            for (feats, coors, edges),labels in tqdm(val_loader):
                pred = model.forward(feats, coors, edges)
                labels=labels.squeeze()
                out=pred.squeeze()
                loss=criterion(out,labels)
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
    batch_size:int=typer.Option(
        1, "--batch-size", "-bs", help="Batch size. Defaults to 10."
    ),
    override:bool=typer.Option(
        False, "--override", help="Override results. Defaults to False"
    ),
    alpha:str=typer.Option(
        4.5, "--alpha", help="Alpha distance to use for labels. Default to 4.5."
    ),
    seed:int=typer.Option(
        0, "--seed", help="Seed to use for training."
    ),
) -> None:
    if (result_folder/Path("summary_dict.json")).exists() and not override :
        print("Not overriding results.")
        return
    elif (result_folder/Path("summary_dict.json")).exists():
        print("Overriding results.")
    result_folder.mkdir(exist_ok=True, parents=True)
    args_dict = {
        "train_folder_path": str(train_folder_path),
        "val_folder_path": str(val_folder_path),
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "result_folder": str(result_folder),
        "positive_weight": positive_weight,
        "batch_size":batch_size,
        "override":override,
        "alpha":alpha,
        "seed":seed,

    }
    if seed>0:
        torch.manual_seed(seed)
    with open(train_folder_path / Path("dict.json")) as f :
        dict_train = json.load(f)
    train_embeddings = torch.load(train_folder_path / Path("embeddings.pt"), weights_only=True)
    train_csv = pd.read_csv("/home/gathenes/paragraph_benchmark/expanded_dataset/train_set.csv")

    with open(val_folder_path / Path("dict.json")) as f :
        dict_val = json.load(f)
    val_embeddings = torch.load(val_folder_path / Path("embeddings.pt"), weights_only=True)
    val_csv = pd.read_csv("/home/gathenes/paragraph_benchmark/expanded_dataset/val_set.csv")



    train_loader = create_dataloader(csv=train_csv,dataset_dict=dict_train, residue_embeddings=train_embeddings, batch_size=batch_size, shuffle=True, alpha=alpha)
    val_loader = create_dataloader(csv=val_csv,dataset_dict=dict_val, residue_embeddings=val_embeddings, batch_size=batch_size, alpha=alpha)
    criterion = nn.BCELoss()

    from graph_model import EGNN_Model
    from torch.utils.data import DataLoader
    model = EGNN_Model(num_feats = 480,graph_hidden_layer_output_dims = [480]*6,linear_hidden_layer_output_dims = [10]*2)

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
    )
    save_plot_path = result_folder / Path("graph_summary_plot.png")
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
    with open(result_folder / Path("graph_summary_dict.json"), 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

if __name__ == "__main__":
    app()

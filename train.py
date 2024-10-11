import json
import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import typer
from create_dataset import create_dataloader
from models import MLP, EarlyStopping
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm

app = typer.Typer(add_completion=False)


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    n_epochs=3,
    model_save_path=Path("/home/gathenes/all_structures/data_high_qual/checkpoint.pt"),
    criterion=nn.BCELoss(),
    mask_prob:float=0,
    lr_strategy:Optional[str]=None,  # Can be "step", "plateau", or "cosine"
):
    # Training loop
    device = torch.device("cpu")
    train_loss_list = []
    val_loss_list = []
    auc_list = []
    ap_list = []
    early_stopping = EarlyStopping(patience=30, path=model_save_path, best_score=0)
    if lr_strategy == "step":
        scheduler = StepLR(optimizer, step_size=20)
    elif lr_strategy == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=10)
    elif lr_strategy == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=50)
    elif lr_strategy:
        raise ValueError(f"Unknown scheduler type: {lr_strategy}")
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        model.train()
        for i, (embedding, labels, len_heavy, len_light) in enumerate(tqdm(train_loader)):
            embedding, labels = embedding.to(device), labels.to(device)
            embedding_list=[]
            label_list = []
            for i in range(len_heavy.shape[-1]):
                heavy, light = len_heavy[i], len_light[i]
                ran = list(range(1,heavy+1))+list(range(heavy+2, heavy+light+2))
                emb=embedding[i][ran]
                lab=labels[i][:heavy+light]
                if random.random()<mask_prob:
                    drop_mask = torch.rand(emb.size(), device=emb.device) >= mask_prob
                    emb=emb * drop_mask.float()
                embedding_list.append(emb)
                label_list.append(lab)

            embedding = torch.cat(embedding_list, dim=0)
            labels = torch.cat(label_list, dim=0)
            optimizer.zero_grad()
            output = model(embedding)
            output=output.view(-1)
            #print(embedding.shape, labels.shape, output.shape)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * embedding.size(0)
        train_loss /= len(train_loader.dataset)
        train_loss_list.append(train_loss)
        val_loss = 0
        all_outputs = []
        all_targets = []
        print(f"Saving train loss and epoch, epoch={epoch}, train_loss={train_loss}")
        if len(val_loader) == 0:
            continue
        with torch.no_grad():
            model.eval()
            for embedding, labels, len_heavy, len_light in val_loader:
                embedding, labels = embedding.to(device), labels.to(device)
                embedding_list=[]
                label_list = []
                for i in range(len_heavy.shape[-1]):
                    heavy, light = len_heavy[i], len_light[i]
                    ran = list(range(1,heavy+1))+list(range(heavy+2, heavy+light+2))
                    embedding_list.append(embedding[i][ran])
                    label_list.append(labels[i][:heavy+light])
                embedding = torch.cat(embedding_list, dim=0)
                labels = torch.cat(label_list, dim=0)
                output = model(embedding)
                output=output.view(-1)
                loss = criterion(output, labels)
                val_loss += loss.item() * embedding.size(0)
                output = output.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                all_outputs.extend(output)
                all_targets.extend(labels)

            val_loss /= len(val_loader.dataset)
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
        if lr_strategy in ["step","cosine"]:
            scheduler.step()  # StepLR scheduler: decrease lr after `step_size` epochs
        elif lr_strategy == "plateau":
            scheduler.step(-ap)
    return train_loss_list, val_loss_list, auc_list, ap_list


def save_plot(train_loss_list, val_loss_list, auc_list, ap_list, save_plot_path):
    n_epochs = len(train_loss_list)

    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # First subplot for Train Loss
    axs[0, 0].plot(range(1, n_epochs + 1), train_loss_list)
    axs[0, 0].set_xlabel("num_epochs")
    axs[0, 0].set_ylabel("Train Loss")
    axs[0, 0].set_title("Train Loss")

    # Second subplot for val Loss
    axs[0, 1].plot(range(1, n_epochs + 1), val_loss_list)
    axs[0, 1].set_xlabel("num_epochs")
    axs[0, 1].set_ylabel("val Loss")
    axs[0, 1].set_title("val Loss")

    # Third subplot for AUC
    axs[1, 0].plot(range(1, n_epochs + 1), auc_list)
    axs[1, 0].set_xlabel("num_epochs")
    axs[1, 0].set_ylabel("AUC")
    axs[1, 0].set_title("AUC")

    # Fourth subplot for AP
    axs[1, 1].plot(range(1, n_epochs + 1), ap_list)
    axs[1, 1].set_xlabel("num_epochs")
    axs[1, 1].set_ylabel("AP")
    axs[1, 1].set_title("Average Precision (AP)")

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.savefig(save_plot_path)


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
    model_name: str = typer.Option(
        "MLP", "--model", "-m", help="Model to use for training."
    ),
    learning_rate: float = typer.Option(
        0.001, "--lr", help="Learning rate to use for training."
    ),
    n_epochs: int = typer.Option(
        1, "--n_epochs", "-n", help="Number of epochs to use for training."
    ),
    result_folder: Path = typer.Option(
        Path("./result/"), "--result_folder", "-r", help="Where to save results."
    ),
    positive_weight: int = typer.Option(
        1, "--pos-weight", help="Weight to give to positive labels."
    ),
    batch_size:int=typer.Option(
        10, "--batch-size", "-bs", help="Batch size. Defaults to 10."
    ),
    lr_strategy:Optional[str]=typer.Option(
        None, "--lr-strat", help="Learning rate strategy to use. Must be 'step', 'plateau' \
            or 'cosine'. Defaults to None."
    ),
    mask_prob:float=typer.Option(
        0, "--mask-prob", help="Probability with which to mask each embedding coefficient. \
            Defaults to 0"
    ),
    dropout_prob:float=typer.Option(
        0, "--dropout", help="Dropout probability. Defaults to 0."
    ),
    dim1:int=typer.Option(
        1000, "--dim1", help="Dimension of first layer. Defaults to 1000."
    ),
    dim2:int=typer.Option(
        1, "--dim2", help="Dimension of second layer. 1 means no second layer. Defaults to 1."
    ),
    batch_norm:bool=typer.Option(
        False, "--batch-norm", help="Whether to use batchnorm or not. Defaults to False."
    ),
) -> None:
    result_folder.mkdir(exist_ok=True, parents=True)
    args_dict = {
        "train_folder_path": str(train_folder_path),
        "val_folder_path": str(val_folder_path),
        "model_name": model_name,
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "result_folder": str(result_folder),
        "positive_weight": positive_weight,
        "dim1":dim1,
        "dim2":dim2,
        "mask_prob":mask_prob,
        "dropout_prob":dropout_prob,
        "lr_strategy":lr_strategy,
        "batch_size":batch_size,
        "batch_norm":batch_norm,
        "flip":flip,
    }
    with open(train_folder_path / Path("dict.json")) as f :
        dict_train = json.load(f)
    train_embeddings = torch.load(train_folder_path / Path("embeddings.pt"), weights_only=True)

    with open(val_folder_path / Path("dict.json")) as f :
        dict_val = json.load(f)
    val_embeddings = torch.load(val_folder_path / Path("embeddings.pt"), weights_only=True)

    train_loader = create_dataloader(dataset_dict=dict_train, residue_embeddings=train_embeddings, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(dataset_dict=dict_val, residue_embeddings=val_embeddings, batch_size=batch_size)
    torch.save(train_loader, result_folder / Path(f'train_dataloader_batchsize_{batch_size}.pkl'))
    torch.save(val_loader, result_folder / Path(f'val_dataloader_batchsize_{batch_size}.pkl'))

    if model_name == "MLP":
        model = MLP(dropout_prob=dropout_prob, dim1=dim1, dim2=dim2, batch_norm=batch_norm)
    else:
        raise ValueError("Model not recognized.")
    if positive_weight == 1:
        criterion = nn.BCELoss()
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([positive_weight], device=torch.device("cpu"))
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model_save_path = result_folder / Path("checkpoint.pt")
    train_loss_list, val_loss_list, auc_list, ap_list = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=n_epochs,
        criterion=criterion,
        model_save_path=model_save_path,
        lr_strategy=lr_strategy,
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
    with open(result_folder / Path("summary_dict.json"), 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

if __name__ == "__main__":
    app()

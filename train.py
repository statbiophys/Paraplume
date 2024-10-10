import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import typer
from models import CNN, MLP, MLP_AA, EarlyStopping
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from tqdm import tqdm

app = typer.Typer(add_completion=False)


def train(
    model,
    train_loader,
    test_loader,
    optimizer,
    n_epochs=3,
    model_save_path=Path("/home/gathenes/all_structures/data_high_qual/checkpoint.pt"),
    criterion=nn.BCELoss(),
):
    # Training loop
    device = torch.device("cpu")
    train_loss_list = []
    test_loss_list = []
    auc_list = []
    ap_list = []
    early_stopping = EarlyStopping(patience=10, path=model_save_path, best_score=0)
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
                embedding_list.append(embedding[i][ran])
                label_list.append(labels[i][:heavy+light])
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
        test_loss = 0
        all_outputs = []
        all_targets = []
        print(f"Saving train loss and epoch, epoch={epoch}, train_loss={train_loss}")
        if len(test_loader) == 0:
            continue
        with torch.no_grad():
            for embedding, labels, len_heavy, len_light in test_loader:
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
                test_loss += loss.item() * embedding.size(0)
                output = output.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                all_outputs.extend(output)
                all_targets.extend(labels)

            test_loss /= len(test_loader.dataset)
            test_loss_list.append(test_loss)

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
    return train_loss_list, test_loss_list, auc_list, ap_list


def save_plot(train_loss_list, test_loss_list, auc_list, ap_list, save_plot_path):
    n_epochs = len(train_loss_list)

    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # First subplot for Train Loss
    axs[0, 0].plot(range(1, n_epochs + 1), train_loss_list)
    axs[0, 0].set_xlabel("num_epochs")
    axs[0, 0].set_ylabel("Train Loss")
    axs[0, 0].set_title("Train Loss")

    # Second subplot for Test Loss
    axs[0, 1].plot(range(1, n_epochs + 1), test_loss_list)
    axs[0, 1].set_xlabel("num_epochs")
    axs[0, 1].set_ylabel("Test Loss")
    axs[0, 1].set_title("Test Loss")

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
    train_loader_path: Path = typer.Argument(
        ...,
        help="Path of trainloader.",
        show_default=False,
    ),
    test_loader_path: Path = typer.Argument(
        ...,
        help="Path of testloader.",
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
    input_dim_x: int = typer.Option(
        256, "--input-dim-x", help="X dimension of the embedding."
    ),
    add_reverse:bool=typer.Option(
        False, "--add-reverse", help="Also add reverse sequences"
    ),
) -> None:
    args_dict = {
        "train_loader_path": str(train_loader_path),
        "test_loader_path": str(test_loader_path),
        "model_name": model_name,
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "result_folder": str(result_folder),
        "positive_weight": positive_weight,
        "input_dim_x": input_dim_x
    }

    result_folder.mkdir(exist_ok=True, parents=True)
    train_loader = torch.load(train_loader_path)
    test_loader = torch.load(test_loader_path)
    if model_name == "MLP":
        model = MLP_AA()
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
    train_loss_list, test_loss_list, auc_list, ap_list = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        n_epochs=n_epochs,
        criterion=criterion,
        model_save_path=model_save_path,
    )
    save_plot_path = result_folder / Path("summary_plot.png")
    save_plot(
        train_loss_list=train_loss_list,
        test_loss_list=test_loss_list,
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

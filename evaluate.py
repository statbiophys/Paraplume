"""Save metrics in dictionary for test set."""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import typer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.nn import Dropout, Linear, Module, ReLU, Sequential, Sigmoid
from torch_dataset import create_dataloader
from tqdm import tqdm
from train import get_outputs
from utils import get_logger

app = typer.Typer(add_completion=False)
log = get_logger()
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=too-many-positional-arguments
# pylint: disable=duplicate-code


def test(
    model: torch.nn.Sequential,
    test_loader: torch.utils.data.dataloader.DataLoader,
    embedding_models_list: List,
    threshold: float,
    best_val_mcc_threshold: float,
    best_val_f1_threshold: float,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Return metrics for the test set.

    Args:
        model (torch.nn.Sequential): Model used for evaluation
        test_loader (torch.utils.data.dataloader.DataLoader): Test data loader
        embedding_models_list (List, optional): List of embeddings to use.
        threshold (float): Youden threshold. Defaults to 0.
        best_val_mcc_threshold (float): Threshold giving best mcc on validation set.\
            Defaults to 0.
        best_val_f1_threshold (float): Threshold giving best f1 on validation set.\
            Defaults to 0.

    Returns:
        auc, ap, f1, mcc, f1_05, mcc_05, best_val_mcc, best_val_f1 \
            (Tuple[float,float,float,float,float,float,float,float]): Metrics to return.
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    all_outputs = np.array([], dtype=np.float32)
    all_targets = np.array([], dtype=np.float32)
    with torch.no_grad():
        for _, (
            embedding,
            labels,
            len_heavy,
            len_light,
        ) in tqdm(enumerate(test_loader)):
            embedding, labels = embedding.to(device), labels.to(device)
            labels, output = get_outputs(
                embedding=embedding,
                labels=labels,
                len_heavy=len_heavy,
                len_light=len_light,
                model=model,
                embedding_models_list=embedding_models_list,
            )
            output = output.view(-1)
            output = output.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            all_outputs = np.concatenate((all_outputs, output))
            all_targets = np.concatenate((all_targets, labels))

    # Calculate the AUC score
    auc = roc_auc_score(all_targets, all_outputs)
    ap = average_precision_score(all_targets, all_outputs)

    all_predictions_f1 = [1 if x >= threshold else 0 for x in all_outputs]
    f1 = f1_score(all_targets, all_predictions_f1)
    all_predictions_mcc = [1 if x >= threshold else 0 for x in all_outputs]
    mcc = matthews_corrcoef(all_targets, all_predictions_mcc)

    all_predictions_f1 = [1 if x >= 0.5 else 0 for x in all_outputs]
    f1_05 = f1_score(all_targets, all_predictions_f1)
    all_predictions_mcc = [1 if x >= 0.5 else 0 for x in all_outputs]
    mcc_05 = matthews_corrcoef(all_targets, all_predictions_mcc)

    all_predictions_mcc = [1 if x >= best_val_mcc_threshold else 0 for x in all_outputs]
    best_val_mcc = matthews_corrcoef(all_targets, all_predictions_mcc)

    all_predictions_f1 = [1 if x >= best_val_f1_threshold else 0 for x in all_outputs]
    best_val_f1 = matthews_corrcoef(all_targets, all_predictions_mcc)

    return auc, ap, f1, mcc, f1_05, mcc_05, best_val_mcc, best_val_f1


@app.command()
def main(
    result_folder: Path = typer.Argument(
        ...,
        help="Path of the result folder.",
        show_default=False,
    ),
    test_folder_path: Path = typer.Argument(
        ...,
        help="Path of testloader.",
        show_default=False,
    ),
    name:str=typer.Option(
        "",
        "--name",
        help="Add name to end of file."
    ),
) -> None:
    """Save metrics in dictionary for test set."""
    model_path = result_folder / Path("checkpoint.pt")
    summary_dict_path = result_folder / Path("summary_dict.json")
    test_dict_path = test_folder_path / Path("dict.json")
    test_embeddings_path = test_folder_path / Path("embeddings.pt")
    log.info("Loading model path", path=model_path.as_posix())
    args_dict = {
        "model_path": str(model_path),
        "test_folder_path": str(test_folder_path),
        "result_folder": str(result_folder),
    }
    log.info("Loading training summary dictionary.", path=summary_dict_path.as_posix())
    with open(summary_dict_path, encoding="utf-8") as f:
        summary_dict = json.load(f)
    batch_size = summary_dict["batch_size"]
    log.info("Loading test loader dictionary.", path=test_dict_path.as_posix())
    with open(test_dict_path, encoding="utf-8") as f:
        dict_test = json.load(f)
    log.info("Loading test embeddings.", path=test_embeddings_path.as_posix())
    test_embeddings = torch.load(test_embeddings_path, weights_only=True)
    log.info("Create test data loader.")
    test_loader = create_dataloader(
        dataset_dict=dict_test,
        embeddings=test_embeddings,
        batch_size=batch_size,
        mode="test",
    )
    log.info("Loading arguments from the training summary dictionary.")
    dims = [int(each) for each in summary_dict["dims"].split(",")]
    dropouts = [float(each) for each in summary_dict["dropouts"].split(",")]
    input_size = int(summary_dict["input_size"])
    embedding_models = summary_dict["embedding_models"]
    if embedding_models == "all":
        embedding_models = "ablang2,igbert,igT5,esm,antiberty,prot-t5"
    embedding_models_list = embedding_models.split(",")
    layers: List[Module] = []
    for i, _ in enumerate(dims):
        if i == 0:
            layers.append(Linear(input_size, dims[i]))
            layers.append(Dropout(dropouts[i]))
            layers.append(ReLU())
        else:
            layers.append(Linear(dims[i - 1], dims[i]))
            layers.append(Dropout(dropouts[i]))
            layers.append(ReLU())
    model = Sequential(Sequential(*layers), Sequential(Linear(dims[-1], 1), Sigmoid()))
    log.info("Loading model", path=model_path.as_posix())
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    log.info("Retrieving results")
    threshold = float(summary_dict["threshold_youden"])
    best_val_mcc_threshold = float(summary_dict["best_val_mcc_threshold"])
    best_val_f1_threshold = float(summary_dict["best_val_f1_threshold"])
    auc, ap, f1, mcc, f1_05, mcc_05, best_val_mcc, best_val_f1 = test(
        model=model,
        test_loader=test_loader,
        embedding_models_list=embedding_models_list,
        threshold=threshold,
        best_val_mcc_threshold=best_val_mcc_threshold,
        best_val_f1_threshold=best_val_f1_threshold,
    )
    args_dict["ap"] = str(ap)
    args_dict["auc"] = str(auc)
    args_dict["f1"] = str(f1)
    args_dict["mcc"] = str(mcc)
    args_dict["f1_05"] = str(f1_05)
    args_dict["mcc_05"] = str(mcc_05)
    args_dict["mcc_best_val"] = str(best_val_mcc)
    args_dict["f1_best_val"] = str(best_val_f1)

    result_dict_path = result_folder / Path(f"{test_folder_path.stem}_results_dict{name}.json")
    log.info("Saving results", path=result_dict_path.as_posix())
    with open(result_dict_path, "w", encoding="utf-8") as json_file:
        json.dump(args_dict, json_file, indent=4)


if __name__ == "__main__":
    app()

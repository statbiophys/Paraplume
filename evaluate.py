import json
from pathlib import Path

import numpy as np
import torch
import typer
from models import CNN, MLP, MLP_AA
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from torch import nn
from tqdm import tqdm

app = typer.Typer(add_completion=False)


def test(
    model,
    test_loader,
    cdr_pm2: bool = False,
    add_reverse:bool=False,
):
    # Training loop
    device = torch.device("cpu")
    all_outputs = []
    all_targets = []
    outputs_and_labels = {}
    with torch.no_grad():
        for i, (
            embedding,
            labels,
            len_heavy,
            len_light,
        ) in tqdm(enumerate(test_loader)):
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
            output=model(embedding)
            output=output.view(-1)

            # Convert the tensors to cpu and then to numpy arrays for AUC and ROC calculation
            output = output.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            all_outputs.extend(output)
            all_targets.extend(labels)

    # Converting lists to numpy arrays for AUC and ROC calculation
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets).astype(int)
    # Calculate the AUC score
    auc = roc_auc_score(all_targets, all_outputs)
    ap = average_precision_score(all_targets, all_outputs)

    return outputs_and_labels, auc, ap


@app.command()
def main(
    model_path: Path = typer.Argument(
        ...,
        help="Path of model.",
        show_default=False,
    ),
    test_loader_path: Path = typer.Argument(
        ...,
        help="Path of testloader.",
        show_default=False,
    ),
    model_type: str = typer.Option(
        "MLP", "--model-type", "-m", help="Model to use for training."
    ),
    result_folder: Path = typer.Option(
        Path("./result/"), "--result_folder", "-r", help="Where to save results."
    ),
    input_dim_x: int = typer.Option(
        256, "--input-dim-x", help="X dimension of the embedding."
    ),
    cdr_pm2:bool=typer.Option(
        False, "--cdr-pm2", help="Whether to evaluate on just cdrs +- 2 or not. Defaults to False."
    ),
    add_reverse:bool=typer.Option(
        False, "--add-reverse", help="Also add reverse sequences"
    ),
) -> None:
    args_dict = {
        "model_path": str(model_path),
        "test_loader_path": str(test_loader_path),
        "result_folder": str(result_folder),
        "cdr-pm2":str(cdr_pm2),
        "input-dim-x":str(input_dim_x),
    }

    result_folder.mkdir(exist_ok=True, parents=True)
    test_loader = torch.load(test_loader_path)
    if model_type=="MLP":
        model = MLP_AA()
    else:
        raise ValueError("No model of this name known.")
    print("LOADING MODEL")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print("RETRIEVING RESULTS")
    outputs_and_labels, auc, ap = test(
        model=model,
        test_loader=test_loader,
        cdr_pm2=cdr_pm2,
        add_reverse=add_reverse,
    )
    args_dict["ap"]=ap
    args_dict["auc"]=auc
    args_dict["detailed_results"]=outputs_and_labels

    print("SAVING RESULTS")
    with open(result_folder / Path("results_dict.json"), "w") as json_file:
        json.dump(args_dict, json_file, indent=4)


if __name__ == "__main__":
    app()

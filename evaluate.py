import json
from pathlib import Path

import numpy as np
import torch
import typer
from create_dataset import create_dataloader
from models import MLP
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
    test_folder_path: Path = typer.Argument(
        ...,
        help="Path of testloader.",
        show_default=False,
    ),
    model_type: str = typer.Option(
        "MLP", "--model-type", "-m", help="Model to use for training."
    ),
    batch_size:int=typer.Option(
        10, "--batch-size", "-bs", help="Batch size. Defaults to 10."
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
    alpha:str=typer.Option(
        4.5, "--alpha", help="Alpha distance to use for labels. Default to 4.5."
    )
) -> None:
    result_folder=model_path.parents[0]
    print(result_folder.as_posix())
    args_dict = {
        "model_path": str(model_path),
        "test_folder_path": str(test_folder_path),
        "result_folder": str(result_folder),
    }
    with open(test_folder_path / Path("dict.json")) as f :
        dict_test = json.load(f)
    test_embeddings = torch.load(test_folder_path / Path("embeddings.pt"), weights_only=True)
    test_loader = create_dataloader(dataset_dict=dict_test, residue_embeddings=test_embeddings, batch_size=batch_size, alpha=alpha)
    #torch.save(test_loader, result_folder / Path(f'test_dataloader_batchsize_{batch_size}.pkl'))

    if model_type=="MLP":
        model = MLP(dim1=dim1, dim2=dim2, batch_norm=batch_norm)
    else:
        raise ValueError("No model of this name known.")
    print("LOADING MODEL")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print("RETRIEVING RESULTS")
    outputs_and_labels, auc, ap = test(
        model=model,
        test_loader=test_loader,
    )
    args_dict["ap"]=ap
    args_dict["auc"]=auc
    args_dict["detailed_results"]=outputs_and_labels

    print("SAVING RESULTS")
    with open(result_folder / Path("results_dict.json"), "w") as json_file:
        json.dump(args_dict, json_file, indent=4)


if __name__ == "__main__":
    app()

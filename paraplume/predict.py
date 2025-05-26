"""Predict paratope probabilities."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from biopandas.pdb import PandasPdb
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from torch.nn import Dropout, Linear, ReLU, Sequential, Sigmoid
from tqdm import tqdm

from paraplume.torch_dataset import create_dataloader
from paraplume.utils import get_embedding, get_logger, read_pdb_to_dataframe

warnings.filterwarnings("ignore")
app = typer.Typer(add_completion=False)
log = get_logger()

THRESHOLD=0.5

def test( # noqa : PLR0913, PLR0915
    model,
    test_loader,
    chains: pd.DataFrame,
    save_path: Path,
    embedding_model_list:list,
    pdb_path: Path = Path("/home/gathenes/all_structures/imgt_renumbered_expanded/"),
    *,
    save=False,
)->tuple:
    """Get metrics for the test set as well as dataframe with predicted paratope proba for all AAs.

    Args:
        model (_type_): Model used for testing.
        test_loader (_type_): Test loader.
        chains (pd.DataFrame): Chains df for test.
        save_path (Path): Where to save results.
        embedding_model_list (list): List of llms used for embedding.
        pdb_path (Path, optional): PDB folder to use for creating PDBs for visualization.\
            Defaults to Path("/home/gathenes/all_structures/imgt_renumbered_expanded/").
        save (bool, optional): Save PDBS. Defaults to False.

    Returns
    -------
        tuple: concatenated dataframe for all pdbs and predicted paratopes, and the metrics.
    """
    device = torch.device("cpu")
    pdb_dict : dict[str,dict]= {}
    total_dataframe = pd.DataFrame()
    ap_list = []
    roc_list = []
    f1_list = []
    mcc_list = []
    total_labs = []
    total_preds = []
    total_preds_bins = []
    with torch.no_grad():
        for _, (
            embedding_raw,
            labels_raw,
            heavy,
            light,
            pdb_code_raw,
            inverse_heavy,
            inverse_light,
        ) in tqdm(enumerate(test_loader)):
            pdb_code = pdb_code_raw[0]
            pdb_dict[pdb_code] = {
                "heavy prediction": {},
                "heavy label": {},
                "light label": {},
                "light prediction": {},
            }
            embedding, labels = embedding_raw.to(device)[0], labels_raw.to(device)[0]
            emb = get_embedding(
                embedding=embedding, embedding_models=embedding_model_list, heavy=heavy, light=light
            )
            output = model(emb)
            output = output.view(-1)
            for i in range(heavy):
                pdb_dict[pdb_code]["heavy prediction"][inverse_heavy[i][0]] = float(
                    output[i].detach().cpu().numpy()
                )
                pdb_dict[pdb_code]["heavy label"][inverse_heavy[i][0]] = float(
                    labels[i].detach().cpu().numpy()
                )
            for i in range(light):
                pdb_dict[pdb_code]["light prediction"][inverse_light[i][0]] = float(
                    output[heavy + i].detach().cpu().numpy()
                )
                pdb_dict[pdb_code]["light label"][inverse_light[i][0]] = float(
                    labels[heavy + i].detach().cpu().numpy()
                )
            data_pdb = read_pdb_to_dataframe(pdb_path / Path(f"{pdb_code}.pdb"))
            light_chain, heavy_chain, antigen_chain = chains.query("pdb==@pdb_code")[
                ["Lchain", "Hchain", "antigen_chain"]
            ].values[0]
            data_pdb_heavy = data_pdb.query("chain_id==@heavy_chain").query("residue_number<129")
            data_pdb_light = data_pdb.query("chain_id==@light_chain").query("residue_number<128")
            data_pdb_antigen = data_pdb.query("chain_id==@antigen_chain")
            data_pdb_heavy["b_factor"] = data_pdb_heavy["IMGT"].map(
                pdb_dict[pdb_code]["heavy prediction"]
            )
            data_pdb_light["b_factor"] = data_pdb_light["IMGT"].map(
                pdb_dict[pdb_code]["light prediction"]
            )
            data_pdb_heavy["occupancy"] = data_pdb_heavy["IMGT"].map(
                pdb_dict[pdb_code]["heavy label"]
            )
            data_pdb_light["occupancy"] = data_pdb_light["IMGT"].map(
                pdb_dict[pdb_code]["light label"]
            )
            data_pdb_antigen["b_factor"] = 0
            data_pdb_antigen["occupancy"] = 0

            new_data_pdb = pd.concat([data_pdb_heavy, data_pdb_light, data_pdb_antigen])

            atomic_df = PandasPdb().read_pdb((pdb_path / Path(f"{pdb_code}.pdb")).as_posix())
            atomic_df = atomic_df.get_model(1)
            atomic_df.df["ATOM"] = new_data_pdb
            if save:
                atomic_df.to_pdb(save_path / f"{pdb_code}.pdb", records=["ATOM"])

            ab_only = pd.concat([data_pdb_heavy, data_pdb_light])
            atomic_df = PandasPdb().read_pdb((pdb_path / Path(f"{pdb_code}.pdb")).as_posix())
            atomic_df = atomic_df.get_model(1)
            atomic_df.df["ATOM"] = ab_only
            if save:
                atomic_df.to_pdb(save_path / f"{pdb_code}_abonly.pdb", records=["ATOM"])

            data_pdb_heavy["chain_type"] = "heavy"
            data_pdb_light["chain_type"] = "light"
            pdb_to_concat = pd.concat([data_pdb_heavy, data_pdb_light]).rename(
                columns={"occupancy": "labels", "b_factor": "prediction"}
            )
            pdb_to_concat["pdb"] = pdb_code
            pdb_to_concat = pdb_to_concat.query("atom_name=='CA'")[
                ["pdb", "chain_type", "residue_name", "IMGT", "labels", "prediction"]
            ]

            preds = pdb_to_concat["prediction"].tolist()
            preds_bin = (pdb_to_concat["prediction"] >= THRESHOLD).astype(int).tolist()

            labs = pdb_to_concat["labels"].tolist()
            ap = average_precision_score(labs, preds)
            roc = roc_auc_score(labs, preds)
            mcc = matthews_corrcoef(labs, preds_bin)
            f1 = f1_score(labs, preds_bin)
            roc_list.append(roc)
            ap_list.append(ap)
            f1_list.append(f1)
            mcc_list.append(mcc)
            total_labs.extend(labs)
            total_preds.extend(preds)
            total_preds_bins.extend(preds_bin)
            total_dataframe = pd.concat([total_dataframe, pdb_to_concat])
    mean_ap = np.mean(ap_list)
    mean_roc = np.mean(roc_list)
    mean_f1 = np.mean(f1_list)
    mean_mcc = np.mean(mcc_list)
    flattened_ap = average_precision_score(total_labs, total_preds)
    flattened_auc = roc_auc_score(total_labs, total_preds)
    flattened_f1 = f1_score(total_labs, total_preds_bins)
    flattened_mcc = matthews_corrcoef(total_labs, total_preds_bins)
    return (
        total_dataframe,
        mean_ap,
        mean_roc,
        mean_f1,
        mean_mcc,
        flattened_ap,
        flattened_auc,
        flattened_f1,
        flattened_mcc,
    )
    # Calculate the AUC score


@app.command()
def main( # noqa : PLR0913, PLR0915
    model_path: Path = typer.Argument( # noqa : B008
        ...,
        help="Path of model.",
        show_default=False,
    ),
    test_folder_path: Path = typer.Argument( # noqa : B008
        ...,
        help="Path of testloader.",
        show_default=False,
    ),
    chains_path: Path = typer.Argument( # noqa : B008
        ...,
        help="Path of test csv.",
        show_default=False,
    ),
    pdb_path: Path = typer.Option( # noqa : B008
        "/home/gathenes/all_structures/imgt_renumbered_expanded",
        "--pdb-folder-path",
        help="Path of pdb folder.",
        show_default=False,
    ),
    save: bool = typer.Option(False, "--save", help="Whether to save pdbs or not."), # noqa : FBT003, FBT001
    name: str = typer.Option("", "--name", help="Add name to end of file."),
) -> None:
    """Predict paratopes given test folder."""
    chains = pd.read_csv(chains_path)
    result_folder = model_path.parents[0]
    print(result_folder.as_posix())
    with (test_folder_path / Path("dict.json")).open(encoding="utf-8") as f:
        dict_test = json.load(f)
    test_embeddings = torch.load(test_folder_path / Path("embeddings.pt"), weights_only=True)
    test_loader = create_dataloader(
        dataset_dict=dict_test, embeddings=test_embeddings, batch_size=1, mode="predict"
    )
    with (result_folder / Path("summary_dict.json")).open(encoding="utf-8") as f:
        summary_dict = json.load(f)
    dims = summary_dict["dims"].split(",")
    dims = [int(each) for each in dims]
    dropouts = summary_dict["dropouts"]
    dropouts = dropouts.split(",")
    dropouts = [float(each) for each in dropouts]
    input_size = int(summary_dict["input_size"])
    embedding_models = summary_dict["embedding_models"]
    if embedding_models == "all":
        embedding_models = "ablang2,igbert,igT5,esm,antiberty,prot-t5"
    embedding_models_list = embedding_models.split(",")
    layers = []
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
    print("LOADING MODEL")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print("RETRIEVING RESULTS")
    save_path = result_folder / Path(f"visualize_{test_folder_path.stem}")
    if save:
        save_path.mkdir(exist_ok=True, parents=True)
    (
        total_dataframe,
        mean_ap,
        mean_roc,
        mean_f1,
        mean_mcc,
        flattened_ap,
        flattened_auc,
        flattened_f1,
        flattened_mcc,
    ) = test(
        model=model,
        test_loader=test_loader,
        chains=chains,
        save_path=save_path,
        pdb_path=pdb_path,
        embedding_model_list=embedding_models_list,
        save=save,
    )
    total_dataframe.to_csv(result_folder / Path(f"prediction_{test_folder_path.stem}{name}.csv"))
    args_dict = {
        "model_path": str(model_path),
        "test_folder_path": str(test_folder_path),
        "result_folder": str(result_folder),
    }
    args_dict["ap"] = mean_ap
    args_dict["roc"] = mean_roc
    args_dict["f1"] = mean_f1
    args_dict["mcc"] = mean_mcc
    args_dict["flattened_ap"] = flattened_ap
    args_dict["flattened_roc"] = flattened_auc
    args_dict["flattened_f1"] = flattened_f1
    args_dict["flattened_mcc"] = flattened_mcc
    result_dict_path = result_folder / Path(f"{test_folder_path.stem}_predictions_dict{name}.json")
    log.info("Saving results", path=result_dict_path.as_posix())
    with result_dict_path.open("w", encoding="utf-8") as json_file:
        json.dump(args_dict, json_file, indent=4)


if __name__ == "__main__":
    app()

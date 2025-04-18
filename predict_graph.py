import json
import warnings
from pathlib import Path

import pandas as pd
import torch
import typer
from biopandas.pdb import PandasPdb
from tqdm import tqdm

from graph_torch_dataset import create_graph_dataloader
from models import EGNN_Model
from utils import get_dim, read_pdb_to_dataframe

warnings.filterwarnings("ignore")
app = typer.Typer(add_completion=False)



def test(
    model,
    test_loader,
    chains: pd.DataFrame,
    pdb_folder_path: str,
    save_path: Path,
    infer_edges:bool=False,
):
    # Training loop
    device = torch.device("cpu")
    pdb_dict = {}
    total_dataframe = pd.DataFrame()
    with torch.no_grad():
        for i, (
            (feats, coors, edges),
            labels,
            pdb_code,
            imgt_heavy,
            imgt_light,
        ) in tqdm(enumerate(test_loader)):
            pdb_code = pdb_code[0]
            pdb_dict[pdb_code] = {
                "heavy prediction": {},
                "heavy label": {},
                "light label": {},
                "light prediction": {},
                "heavy convex_hull": {},
                "light convex_hull": {},
                "heavy distances": {},
                "light distances": {},
            }
            if infer_edges:
                output= model(feats, coors, edges=None)
            else:
                output = model(feats, coors, edges)
            output=torch.sigmoid(output)
            output=output.squeeze()
            labels=labels.squeeze()
            imgt_heavy = [each[0] for each in imgt_heavy]
            imgt_light = [each[0] for each in imgt_light]
            for i,each in enumerate(imgt_heavy):
                pdb_dict[pdb_code]["heavy prediction"][each] = float(
                    output[i].detach().cpu().numpy()
                )
                pdb_dict[pdb_code]["heavy label"][each] = float(
                    labels[i].detach().cpu().numpy()
                )

            for i,each in enumerate(imgt_light):
                pdb_dict[pdb_code]["light prediction"][each] = float(
                    output[len(imgt_heavy) + i].detach().cpu().numpy()
                )
                pdb_dict[pdb_code]["light label"][each] = float(
                    labels[len(imgt_heavy) + i].detach().cpu().numpy()
                )

            pdb_path = f"{pdb_folder_path}/{pdb_code}.pdb"
            data_pdb = read_pdb_to_dataframe(pdb_path)
            light_chain, heavy_chain, antigen_chain = chains.query("pdb==@pdb_code")[
                ["Lchain", "Hchain", "antigen_chain"]
            ].values[0]
            data_pdb_heavy = data_pdb.query("chain_id==@heavy_chain").query(
                "IMGT.isin(@imgt_heavy)"
            )
            data_pdb_light = data_pdb.query("chain_id==@light_chain").query(
                "IMGT.isin(@imgt_light)"
            )
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


            new_data_pdb = pd.concat([data_pdb_heavy, data_pdb_light, data_pdb_antigen])
            atomic_df = PandasPdb().read_pdb(pdb_path)
            atomic_df = atomic_df.get_model(1)
            if len(atomic_df.df["ATOM"]) == 0:
                raise ValueError(f"No model found for index: {1}")
            atomic_df.df["ATOM"] = new_data_pdb.query("record_name=='ATOM'")
            atomic_df.to_pdb(save_path / f"{pdb_code}.pdb", records=["ATOM"])

            data_pdb_heavy["chain_type"] = "heavy"
            data_pdb_light["chain_type"] = "light"
            pdb_to_concat = pd.concat([data_pdb_heavy, data_pdb_light]).rename(
                columns={"occupancy": "labels", "b_factor": "prediction"}
            )
            pdb_to_concat["pdb"] = pdb_code
            pdb_to_concat = pdb_to_concat.query("atom_name=='CA'")[
                [
                    "pdb",
                    "chain_type",
                    "residue_name",
                    "IMGT",
                    "labels",
                    "prediction",
                ]
            ]
            total_dataframe = pd.concat([total_dataframe, pdb_to_concat])

    return total_dataframe
    # Calculate the AUC score


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
    chains_path: Path = typer.Argument(
        ...,
        help="Path of test csv.",
        show_default=False,
    ),
    infer_edges: bool = typer.Option(
        False, "--infer-edges", help="Infer edges instead of using sparse graph."
    ),
    pdb_folder_path:Path=typer.Option(
        "/home/gathenes/all_structures/imgt_renumbered_expanded",
        "--pdb-folder-path",
        help="Path of pdb folder.",
        show_default=False,
    ),
    name:str=typer.Option(
        "","--name", help="Extension to the files",)
) -> None:

    chains = pd.read_csv(chains_path)
    result_folder = model_path.parents[0]
    print(result_folder.as_posix())
    with open(result_folder / Path("graph_summary_dict.json"), encoding="utf-8") as f:
        summary_dict = json.load(f)


    embedding_models = summary_dict["embedding_models"]
    if embedding_models=='all':
        embedding_models="ablang2,igbert,igT5,esm,antiberty,prot-t5"
    embedding_models_list = embedding_models.split(",")

    with open(test_folder_path / Path("dict.json"), encoding="utf-8") as f:
        dict_test = json.load(f)
    test_embeddings = torch.load(
        test_folder_path / Path("embeddings.pt"), weights_only=True
    )
    test_loader = create_graph_dataloader(
        dataset_dict=dict_test,
        embeddings=test_embeddings,
        pdb_folder_path=pdb_folder_path,
        csv=chains,
        embedding_models=embedding_models_list,
    )


    infer_edges = summary_dict["infer-edges"]
    if infer_edges:
        edge_dim = 0
    else:
        edge_dim = 1
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
    )

    print("LOADING MODEL")
    model.load_state_dict(torch.load(model_path, weights_only=True,map_location=torch.device('cpu')))
    model.eval()
    print("RETRIEVING RESULTS")
    save_path = result_folder / Path(f"visualize_{name}")
    save_path.mkdir(exist_ok=True, parents=True)
    total_dataframe = test(
        model=model,
        test_loader=test_loader,
        chains=chains,
        pdb_folder_path=pdb_folder_path,
        save_path=save_path,
    )
    total_dataframe.to_csv(result_folder / Path(f"prediction_{name}.csv"))


if __name__ == "__main__":
    app()

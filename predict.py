import json
import warnings
from pathlib import Path

import pandas as pd
import torch
import typer
from biopandas.pdb import PandasPdb
from torch.nn import BatchNorm1d, Dropout, Linear, ReLU, Sequential, Sigmoid
from torch_dataset import create_dataloader
from tqdm import tqdm
from utils import read_pdb_to_dataframe

warnings.filterwarnings("ignore")
app = typer.Typer(add_completion=False)

def test(
    model,
    test_loader,
    chains:pd.DataFrame,
    save_path: Path,
    big_embedding:bool=True,
    pdb_path : Path = Path("/home/gathenes/all_structures/imgt_renumbered_expanded/"),
):
    # Training loop
    device = torch.device("cpu")
    pdb_dict = {}
    total_dataframe=pd.DataFrame()
    with torch.no_grad():
        for i, (
            embedding,
            labels,
            heavy,
            light,
            pdb_code,
            inverse_heavy,
            inverse_light,
            convex_hull,
            distances,
        ) in tqdm(enumerate(test_loader)):
            pdb_code=pdb_code[0]
            pdb_dict[pdb_code]={"heavy prediction":{},"heavy label":{}, "light label":{}, "light prediction":{}, "heavy convex_hull":{}, "light convex_hull":{}, "heavy distances":{}, "light distances":{}}
            embedding, labels = embedding.to(device), labels.to(device)
            embedding, labels=embedding[0], labels[0]
            ran = list(range(1,heavy+1))+list(range(heavy+2, heavy+light+2))
            emb=embedding[ran,:2048]
            if big_embedding :
                ran2 = list(range(1,heavy+1))+list(range(heavy+4, heavy+light+4))
                emb2=embedding[ran2,2048:]
                emb=torch.cat([emb,emb2], dim=1)
            output=model(emb)
            output=output.view(-1)
            for i in range(heavy):
                pdb_dict[pdb_code]["heavy prediction"][inverse_heavy[i][0]]=float(output[i].detach().cpu().numpy())
                pdb_dict[pdb_code]["heavy label"][inverse_heavy[i][0]]=float(labels[i].detach().cpu().numpy())
                pdb_dict[pdb_code]["heavy convex_hull"][inverse_heavy[i][0]]=float(convex_hull[0][i].detach().cpu().numpy())
                pdb_dict[pdb_code]["heavy distances"][inverse_heavy[i][0]]=float(distances[0][i].detach().cpu().numpy())
            for i in range(light):
                pdb_dict[pdb_code]["light prediction"][inverse_light[i][0]]=float(output[heavy+i].detach().cpu().numpy())
                pdb_dict[pdb_code]["light label"][inverse_light[i][0]]=float(labels[heavy+i].detach().cpu().numpy())
                pdb_dict[pdb_code]["light convex_hull"][inverse_light[i][0]]=float(convex_hull[0][heavy+i].detach().cpu().numpy())
                pdb_dict[pdb_code]["light distances"][inverse_light[i][0]]=float(distances[0][heavy+i].detach().cpu().numpy())
            data_pdb= read_pdb_to_dataframe((pdb_path / Path(f"{pdb_code}.pdb")).as_posix())
            light_chain, heavy_chain, antigen_chain = chains.query("pdb==@pdb_code")[["Lchain", "Hchain", "antigen_chain"]].values[0]
            data_pdb_heavy=data_pdb.query("chain_id==@heavy_chain").query("residue_number<129")
            data_pdb_light=data_pdb.query("chain_id==@light_chain").query("residue_number<128")
            data_pdb_antigen=data_pdb.query("chain_id==@antigen_chain")
            data_pdb_heavy["b_factor"]=data_pdb_heavy["IMGT"].map(pdb_dict[pdb_code]["heavy prediction"])
            data_pdb_light["b_factor"]=data_pdb_light["IMGT"].map(pdb_dict[pdb_code]["light prediction"])
            data_pdb_heavy["occupancy"]=data_pdb_heavy["IMGT"].map(pdb_dict[pdb_code]["heavy label"])
            data_pdb_light["occupancy"]=data_pdb_light["IMGT"].map(pdb_dict[pdb_code]["light label"])
            data_pdb_antigen["b_factor"]=0
            data_pdb_antigen["occupancy"]=0

            new_data_pdb = pd.concat([data_pdb_heavy, data_pdb_light, data_pdb_antigen])

            atomic_df = PandasPdb().read_pdb((pdb_path / Path(f"{pdb_code}.pdb")).as_posix())
            atomic_df = atomic_df.get_model(1)
            atomic_df.df["ATOM"] = new_data_pdb
            atomic_df.to_pdb(save_path / f"{pdb_code}.pdb", records = ["ATOM"])

            data_pdb_heavy["convex_hull"]=data_pdb_heavy["IMGT"].map(pdb_dict[pdb_code]["heavy convex_hull"])
            data_pdb_light["convex_hull"]=data_pdb_light["IMGT"].map(pdb_dict[pdb_code]["light convex_hull"])
            data_pdb_heavy["distances"]=data_pdb_heavy["IMGT"].map(pdb_dict[pdb_code]["heavy distances"])
            data_pdb_light["distances"]=data_pdb_light["IMGT"].map(pdb_dict[pdb_code]["light distances"])


            data_pdb_heavy["chain_type"]="heavy"
            data_pdb_light["chain_type"]="light"
            pdb_to_concat = pd.concat([data_pdb_heavy,data_pdb_light]).rename(columns={"occupancy":"labels", "b_factor":"prediction"})
            pdb_to_concat["pdb"]=pdb_code
            pdb_to_concat = pdb_to_concat.query("atom_name=='CA'")[["pdb","chain_type","residue_name","IMGT","labels", "prediction", "convex_hull", "distances"]]

            total_dataframe=pd.concat([total_dataframe, pdb_to_concat])


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
    pdb_path:Path=typer.Option(
        "/home/gathenes/all_structures/imgt_renumbered_expanded",
        "--pdb-folder-path",
        help="Path of pdb folder.",
        show_default=False,
    )
) -> None:
    chains=pd.read_csv(chains_path)
    result_folder=model_path.parents[0]
    print(result_folder.as_posix())
    with open(test_folder_path / Path("dict.json"), encoding="utf-8") as f :
        dict_test = json.load(f)
    test_embeddings = torch.load(test_folder_path / Path("embeddings.pt"), weights_only=True)
    test_loader = create_dataloader(dataset_dict=dict_test, embeddings=test_embeddings, batch_size=1, alpha=4.5, mode="predict")
    with open(result_folder / Path("summary_dict.json"), encoding="utf-8") as f:
        summary_dict = json.load(f)
    dims=summary_dict["dims"].split(",")
    dims=[int(each) for each in dims]
    dropouts=summary_dict["dropouts"]
    dropouts=dropouts.split(",")
    dropouts=[float(each) for each in dropouts]
    batchnorm=summary_dict["batchnorm"]
    big_embedding=summary_dict["big_embedding"]
    if big_embedding :
        input_size = 2528
    else:
        input_size = 2048
    layers=[]
    for i, _ in enumerate(dims):
        if i==0:
            layers.append(Linear(input_size, dims[i]))
            layers.append(Dropout(dropouts[i]))
            if batchnorm:
                layers.append(BatchNorm1d)
            layers.append(ReLU())

        else :
            layers.append(Linear(dims[i-1], dims[i]))
            layers.append(Dropout(dropouts[i]))
            if batchnorm:
                layers.append(BatchNorm1d)
            layers.append(ReLU())
    model=Sequential(Sequential(*layers), Sequential(Linear(dims[-1],1),Sigmoid()))
    print("LOADING MODEL")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print("RETRIEVING RESULTS")
    save_path=result_folder / Path(f"visualize_{test_folder_path.stem}")
    save_path.mkdir(exist_ok=True, parents=True)
    total_dataframe=test(
        model=model,
        test_loader=test_loader,
        chains=chains,
        save_path=save_path,
        pdb_path=pdb_path,
    )
    total_dataframe.to_csv(result_folder/Path(f"prediction_{test_folder_path.stem}.csv"))


if __name__ == "__main__":
    app()

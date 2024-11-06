import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from tqdm import tqdm
from utils import format_pdb

sys.setrecursionlimit(100000)


def rec_dd():
    return defaultdict(rec_dd)

import typer

app = typer.Typer(add_completion=False)

def rec_dd():
    return defaultdict(rec_dd)

def add_convex_hull_column(df, mean=False):
    df_copy = deepcopy(df)
    len_df = len(df_copy)
    # Initialize the convex hull column with NaN (not part of any convex hull yet)
    df_copy["convex_hull"] = np.nan
    # Extract the 3D coordinates as a numpy array
    points = df_copy[["x", "y", "z"]].to_numpy()
    k = 1
    total_df = pd.DataFrame()
    while len(points) > 3:
        # Need at least 4 points to form a convex hull in 3D
        # Compute the convex hull for the current set of points
        hull = ConvexHull(points)
        hull_indices = hull.vertices
        # Mark these points in the DataFrame with the current iteration k

        df_copy.loc[df_copy.index[hull_indices], "convex_hull"] = k

        # Remove the points on the current convex hull
        points = np.delete(points, hull_indices, axis=0)
        total_df = pd.concat([total_df, df_copy.loc[df_copy.index[hull_indices]]])

        df_copy = df_copy.drop(df_copy.index[hull_indices])

        k += 1
    if mean:
        res_to_ch = total_df.set_index("Atom_Num")["convex_hull"].to_dict()
        df['convex_hull'] = df['Atom_Num'].map(res_to_ch).fillna(k).astype(float)
    else:
        res_to_ch = total_df.set_index("Res_Num")["convex_hull"].to_dict()
        df['convex_hull'] = df['Res_Num'].map(res_to_ch).fillna(k).astype(float)
    return df

@app.command()
def main(
    dataset_dict_path: Path = typer.Argument(
        ...,
        help="Path of dataset_dict to use for pdb list.",
        show_default=False,
    ),
    csv_path:Path=typer.Argument(
        ...,
        help="Path of csv to use.",
        show_default=False,
    )
) -> None:
    with open (dataset_dict_path) as f:
        dataset_dict=json.load(f)
    for i,value_dict in tqdm(dataset_dict.items()):
        pdb=value_dict["pdb_code"]
        Hchain,Lchain = pd.read_csv(csv_path).query("pdb==@pdb")[["Hchain","Lchain"]].values[0]
        chains=[Hchain,Lchain]
        df_pdb = format_pdb(f"/home/gathenes/all_structures/imgt/{pdb}.pdb").query("Chain.isin(@chains)").astype({"x":float,"y":float,"z":float})
        df_pdb["IMGT"] = df_pdb["Res_Num"].str.replace(r'[a-zA-Z]$', '', regex=True).astype(int)
        df_pdb=df_pdb.query("IMGT<129")
        df_pdb = add_convex_hull_column(df_pdb, mean=True)

        df_pdb_heavy = df_pdb.query("Chain==@Hchain")
        df_pdb_heavy = df_pdb_heavy.groupby("Res_Num", as_index=False)["convex_hull"].mean()
        ch_heavy_dict=df_pdb_heavy.set_index('Res_Num')['convex_hull'].to_dict()
        heavy_numbers=value_dict["H_id numbers"]
        heavy_convex_hull=[]
        for each in heavy_numbers:
            heavy_convex_hull.append(ch_heavy_dict[each])
        dataset_dict[i]["H_id convex_hull mean"]=heavy_convex_hull

        df_pdb_light = df_pdb.query("Chain==@Lchain")
        df_pdb_light=df_pdb_light.query("IMGT<128")
        df_pdb_light = df_pdb_light.groupby("Res_Num", as_index=False)["convex_hull"].mean()
        ch_light_dict=df_pdb_light.set_index('Res_Num')['convex_hull'].to_dict()
        light_numbers=value_dict["L_id numbers"]
        light_convex_hull=[]
        for each in light_numbers:
            light_convex_hull.append(ch_light_dict[each])
        dataset_dict[i]["L_id convex_hull mean"]=light_convex_hull



    with open(dataset_dict_path.parents[0]/Path("dict.json"), "w") as f:
        json.dump(dataset_dict, f)



if __name__ == "__main__":
    app()

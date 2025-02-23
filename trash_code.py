def add_convex_hull_column(df: pd.DataFrame):
    """Add convex hull column to dataframe.

    Args:
        df (pd.DataFrame): Dataframe representing antibody.

    Returns:
        df (pd.DataFrame): Dataframe representing antibody with convex hull info per amino acid.
    """
    df_copy = deepcopy(df)
    # Initialize the convex hull column with NaN (not part of any convex hull yet)
    df_copy["convex_hull"] = np.nan
    # Extract the 3D coordinates as a numpy array
    points = df_copy[["x_coord", "y_coord", "z_coord"]].to_numpy()
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

    res_to_ch = total_df.set_index("IMGT")["convex_hull"].to_dict()
    df["convex_hull"] = df["IMGT"].map(res_to_ch).fillna(k).astype(int)
    return df

def create_convex_hull(pdb_list_path:Path, pdb_folder_path:Path, dataset_dict:Dict):
    """Create convex hull value in dictionary.

    Args:
        pdb_list_path (Path): Path of csv of pdbs and their chains.
        pdb_folder_path (Path): Folder of pdb files.
        dataset_dict (Dict): Dictionary of pdbs and information.

    Returns:
        dataset_dict(Dict): Dictionary with convex hull information.
    """
    for i, value_dict in tqdm(dataset_dict.items()):
        pdb = value_dict["pdb_code"]
        Hchain, Lchain = pd.read_csv(pdb_list_path).query("pdb==@pdb")[["Hchain", "Lchain"]].values[0]
        chains = [Hchain, Lchain]
        df_pdb = (
            read_pdb_to_dataframe(f"{pdb_folder_path}/{pdb}.pdb")
            .query("chain_id.isin(@chains) and residue_number<129")
        )
        df_pdb = add_convex_hull_column(df_pdb)

        df_pdb_heavy = df_pdb.query("chain_id==@Hchain")
        ch_heavy_dict = df_pdb_heavy.set_index("IMGT")["convex_hull"].to_dict()
        heavy_numbers = value_dict["H_id numbers"]
        heavy_convex_hull = []
        for each in heavy_numbers:
            heavy_convex_hull.append(ch_heavy_dict[each])
        dataset_dict[i]["H_id convex_hull"] = heavy_convex_hull

        df_pdb_light = df_pdb.query("chain_id==@Lchain")
        df_pdb_light = df_pdb_light.query("residue_number<128")
        ch_light_dict = df_pdb_light.set_index("IMGT")["convex_hull"].to_dict()
        light_numbers = value_dict["L_id numbers"]
        light_convex_hull = []
        for each in light_numbers:
            light_convex_hull.append(ch_light_dict[each])
        dataset_dict[i]["L_id convex_hull"] = light_convex_hull
    return dataset_dict

def create_convex_hull_mean(pdb_list_path:Path, pdb_folder_path:Path, dataset_dict:Dict):
    """Create mean convex hull value in dictionary.

    Args:
        pdb_list_path (Path): Path of csv of pdbs and their chains.
        pdb_folder_path (Path): Folder of pdb files.
        dataset_dict (Dict): Dictionary of pdbs and information.

    Returns:
        dataset_dict(Dict): Dictionary with mean convex hull information.
    """
    for i, value_dict in tqdm(dataset_dict.items()):
        pdb = value_dict["pdb_code"]
        Hchain, Lchain = pd.read_csv(pdb_list_path).query("pdb==@pdb")[["Hchain", "Lchain"]].values[0]
        chains = [Hchain, Lchain]
        df_pdb = (
            read_pdb_to_dataframe(f"{pdb_folder_path}/{pdb}.pdb")
            .query("chain_id.isin(@chains) and residue_number<129")
        )
        df_pdb = add_convex_hull_column(df_pdb)

        df_pdb_heavy = df_pdb.query("chain_id==@Hchain")
        df_pdb_heavy = df_pdb_heavy.groupby("IMGT", as_index=False)["convex_hull"].mean()
        ch_heavy_dict = df_pdb_heavy.set_index("IMGT")["convex_hull"].to_dict()
        heavy_numbers = value_dict["H_id numbers"]
        heavy_convex_hull = []
        for each in heavy_numbers:
            heavy_convex_hull.append(ch_heavy_dict[each])
        dataset_dict[i]["H_id convex_hull mean"] = heavy_convex_hull

        df_pdb_light = df_pdb.query("chain_id==@Lchain and residue_number<128")
        df_pdb_light = df_pdb_light.groupby("IMGT", as_index=False)["convex_hull"].mean()
        ch_light_dict = df_pdb_light.set_index("IMGT")["convex_hull"].to_dict()
        light_numbers = value_dict["L_id numbers"]
        light_convex_hull = []
        for each in light_numbers:
            light_convex_hull.append(ch_light_dict[each])
        dataset_dict[i]["L_id convex_hull mean"] = light_convex_hull
    return dataset_dict

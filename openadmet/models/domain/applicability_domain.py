import pandas as pd
from rdkit import DataStructs
import numpy as np
from useful_rdkit_utils.descriptors import smi2morgan_fp
import datamol as dm
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from openadmet.models.anvil.specification import DataSpec


def _calculate_top_k_similarity(train_fp, test_fp, top_k=1):
    """
    Calculate the Tanimoto similarity between two sets of fingerprints.
    Adapted from https://github.com/PatWalters/useful_rdkit_utils/blob/master/useful_rdkit_utils/descriptors.py

    Parameters
    ----------
    train_fp : pd.DataFrame
        DataFrame containing training fingerprints.
    test_fp : pd.DataFrame
        DataFrame containing test fingerprints.
    top_k : int, optional
        Number of top similar fingerprints to consider in average, if k=1 corresponds to maximum similarity.

    """
    similarities = []
    for fp in test_fp:
        # calculate Tanimoto similarity to all molecules in training set
        sim_list = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fp))
        sim_array = np.array(sim_list)
        # get top k similarities
        top_k_similarities = np.sort(sim_array)[-top_k:]
        # calculate average similarity
        avg_similarity = np.mean(top_k_similarities)
        similarities.append(avg_similarity)
    return np.array(similarities)


def calculate_ad(
    train_smiles, test_smiles, threshold=0.35, top_k=1, radius=2, nBits=2048
):
    if isinstance(train_smiles, pd.Series):
        train_smiles = train_smiles.values

    if isinstance(test_smiles, pd.Series):
        test_smiles = test_smiles.values

    with dm.without_rdkit_log():
        train_fps = [
            smi2morgan_fp(smile, radius=radius, nBits=nBits) for smile in train_smiles
        ]
        test_fps = [
            smi2morgan_fp(smile, radius=radius, nBits=nBits) for smile in test_smiles
        ]

    similarities = _calculate_top_k_similarity(train_fps, test_fps, top_k=top_k)
    ad_flags = similarities >= threshold
    return ad_flags, similarities


def tantimoto_similarity_from_anvil(
    data_path,
    anvil_dir,
    test_smiles_col="SMILES",
    threshold=0.35,
    top_k=1,
    do_plot=True,
    plot_path="ad_boxplot.png",
    radius=2,
    nBits=2048,
):
    if not Path(anvil_dir).exists():
        raise ValueError(f"Anvil directory {anvil_dir} does not exist.")

    if not Path(data_path).exists():
        raise ValueError(f"Test data file {data_path} does not exist.")

    train_data_path_csv = Path(anvil_dir) / "data/X_train.csv"
    # find what the smiles column was from anvil training spec

    data_spec = Path(anvil_dir) / "recipe_components/data.yaml"
    if not data_spec.exists():
        raise FileNotFoundError(f"Model path {model_dir} does not contain data.yaml")
    # Load the data specification
    data = DataSpec.from_yaml(data_spec)
    x_col = data.input_col

    train_df = pd.read_csv(train_data_path_csv)
    anvil_train_smiles = train_df[x_col]

    test_df = pd.read_csv(data_path)
    test_smiles = test_df[test_smiles_col]

    ad_flags, similarities = calculate_ad(
        anvil_train_smiles,
        test_smiles,
        threshold=threshold,
        top_k=top_k,
        radius=radius,
        nBits=nBits,
    )

    if do_plot:
        fig = plt.figure(figsize=(8, 6))

        sns.ecdfplot(similarities)

        plt.axvline(
            x=threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"AD Threshold ({threshold:.2f})",
        )

        plt.axvspan(xmin=0, xmax=threshold, color="red", alpha=0.1, label="Outside AD")

        plt.axvspan(xmin=threshold, xmax=1, color="green", alpha=0.1, label="Within AD")
        pct_in = np.sum(ad_flags) / len(ad_flags) * 100
        pct_out = 100 - pct_in

        # Position: Top-left of the plot area
        x_pos = 0.05
        y_pos_in = 0.95
        y_pos_out = 0.88
        # give text white background for readability
        plt.text(
            x=x_pos,
            y=y_pos_in,
            s=f"Within AD: {pct_in:.1f}%",
            color="green",
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment="top",
            backgroundcolor="white",
        )
        plt.text(
            x=x_pos,
            y=y_pos_out,
            s=f"Outside AD: {pct_out:.1f}%",
            color="red",
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment="top",
            backgroundcolor="white",
        )

        plt.ylabel("Proportion")
        plt.xlabel(
            f"Top k (k={top_k}) Avg TanimotoSim MorganFP[r={radius}, nBits={nBits}]"
        )
        # Save the figure
        plt.savefig(plot_path, bbox_inches="tight")

        return ad_flags, similarities, fig

    return ad_flags, similarities, None

import pandas as pd 
from rdkit import DataStructs
import numpy as np
from useful_rdkit_utils.descriptors import smi2morgan_fp
import datamol as dm
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

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
        sim_array  = np.array(sim_list)
        # get top k similarities
        top_k_similarities = np.sort(sim_array)[-top_k:]
        # calculate average similarity
        avg_similarity = np.mean(top_k_similarities)
        similarities.append(avg_similarity)
    return np.array(similarities)


def calculate_ad(train_smiles, test_smiles, threshold=0.35, top_k=1):
    if isinstance(train_smiles, pd.Series):
        train_smiles = train_smiles.values
    
    if isinstance(test_smiles, pd.Series):
        test_smiles = test_smiles.values

    with dm.without_rdkit_log():
        train_fps = [smi2morgan_fp(smile, radius=2, nBits=2048) for smile in train_smiles]
        test_fps = [smi2morgan_fp(smile, radius=2, nBits=2048) for smile in test_smiles]

    similarities = _calculate_top_k_similarity(train_fps, test_fps, top_k=top_k)
    ad_flags = similarities >= threshold
    return ad_flags, similarities



def ad_from_anvil(test_data_path, anvil_dir, test_smiles_col='SMILES', train_smiles_col='SMILES', threshold=0.35, top_k=1, do_plot=True, plot_path='ad_boxplot.png'):
    if not Path(anvil_dir).exists():
        raise ValueError(f"Anvil directory {anvil_dir} does not exist.")

    if not Path(test_data_path).exists():
        raise ValueError(f"Test data file {test_data_path} does not exist.")

    train_data_path_csv = Path(anvil_dir) / "data/X_train.csv"
    train_df = pd.read_csv(train_data_path_csv)
    anvil_train_smiles = train_df[train_smiles_col]

    test_df = pd.read_csv(test_data_path)
    test_smiles = test_df[test_smiles_col]

    ad_flags, similarities = calculate_ad(anvil_train_smiles, test_smiles, threshold=threshold, top_k=top_k)
    # make boxplot
    if do_plot:
        sns.boxplot(x=ad_flags, y=similarities)
        sns.stripplot(x=ad_flags, y=similarities, color='black', alpha=0.3)
        plt.xlabel('Within AD')
        plt.ylabel('Tanimoto Similarity')
        plt.title('AD Similarity Distribution')
        plt.savefig(plot_path)
        plt.close()
        return ad_flags, similarities, plot_path
    return ad_flags, similarities
    

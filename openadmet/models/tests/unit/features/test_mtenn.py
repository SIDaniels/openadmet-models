import numpy as np
import pandas as pd
import pytest
import torch

from openadmet.models.features.mtenn import MTENNDataset, MTENNFeaturizer


@pytest.fixture
def mock_complex_features(mocker):
    """
    Patch MTENN complex loading with lightweight synthetic tensors.

    We mock `_load_complexes` to avoid needing actual PDB/SDF files and heavy RDKit/OpenBabel parsing.
    This isolates the MTENNDataset and MTENNFeaturizer logic, allowing us to verify data structuring
    and tensor shapes without file I/O overhead.
    """
    pos = torch.randn(5, 3)
    z = torch.tensor([6, 6, 8, 1, 1], dtype=torch.int32)
    b = torch.ones(5, dtype=torch.float32)
    lig_mask = torch.tensor([False, False, True, True, True], dtype=torch.bool)

    def _mock_load_complexes(complexes, ligand_resname, ignore_h=True):
        n = len(complexes)
        return (
            [pos.clone() for _ in range(n)],
            [z.clone() for _ in range(n)],
            [b.clone() for _ in range(n)],
            [lig_mask.clone() for _ in range(n)],
        )

    mocker.patch.object(
        MTENNDataset, "_load_complexes", side_effect=_mock_load_complexes
    )

    return pos, z, b, lig_mask


def test_mtenn_dataset(mock_complex_features):
    """
    Validate that MTENNDataset correctly constructs data items from complex features.

    This ensures that the dataset class properly organizes positions, atomic numbers,
    and masks into the dictionary format expected by MTENN models.
    """
    pos, z, b, lig_mask = mock_complex_features
    dataset = MTENNDataset(
        ["complex_a", "complex_b"],
        np.asarray([42, 43]),
        ligand_resname="LIG",
        ignore_h=True,
    )

    assert len(dataset) == 2
    feats = dataset[0]

    assert feats["Y"] == 42
    assert feats["pos"].shape == pos.shape
    assert torch.equal(feats["Z"], z)
    assert torch.equal(feats["B"], b)
    assert torch.equal(feats["lig_mask"], lig_mask)


def test_mtenn_featurizer(mock_complex_features):
    """
    Validate the MTENNFeaturizer high-level interface.

    This checks that the featurizer correctly instantiates the dataset and data loader,
    returning formatted batches ready for training.
    """
    ft = MTENNFeaturizer(ligand_resname="LIG", ignore_h=True, batch_size=2, n_jobs=0)
    dataloader, idx, scaler, dataset = ft.featurize(
        ["complex_a", "complex_b"], pd.Series([42.0, 43.0])
    )

    assert len(dataset) == 2
    assert len(dataloader) == 1
    assert np.array_equal(idx, np.array([0, 1]))
    assert scaler is None

    feats, y = next(iter(dataloader))
    assert y.shape == (2, 1)
    assert feats[0]["pos"].shape[1] == 3
    assert feats[0]["z"].ndim == 1
    assert feats[0]["lig"].dtype == torch.bool

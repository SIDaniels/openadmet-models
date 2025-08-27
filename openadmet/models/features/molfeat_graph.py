from collections.abc import Iterable
from typing import Any, ClassVar

import datamol as dm
import numpy as np
from molfeat.trans.graph.adj import PYGGraphTransformer
from molfeat.calc.atom import AtomCalculator
from molfeat.calc.bond import EdgeMatCalculator
from pydantic import Field
from torch.utils.data import DataLoader, Dataset

from openadmet.models.features.feature_base import MolfeatFeaturizer, featurizers



class PyGGraphDataset(Dataset):
    def __init__(self, smiles: Iterable[str], y: Iterable[Any], ):
        self.smiles = smiles
        self.y = y
        self.transformer -=


    def __len__(self):
        return len(self.smiles)


    def __getitem__(self, idx):


@featurizers.register("PyGGraphFeaturizer")
class PyGGraphFeaturizer(MolfeatFeaturizer):
    """
    Fingerprint featurizer for molecules, relies on molfeat backend
    """

    type: ClassVar[str] = "GraphFeaturizer"
    self_loop: bool = Field(
        False, title="Self-loop", description="Whether to add self-loops to atoms"
    )
    bond_self_loop: bool = Field(
        False, title="Bond self-loop", description="Whether to add self-loops to bonds"
    )
    atom_self_loop: bool = Field(
        False, title="Atom self-loop", description="Whether to add self-loops to atoms"
    )
    canonical_atom_order: bool = Field(
        False, title="Canonical atom order", description="Whether to use canonical atom order"
    )
    bond_featurizer = Literal["EdgeMatCalculator"] = Field(

    _transformer: PYGGraphTransformer


    def _prepare(self):
        """
        Prepare the featurizer
        """
        self._transformer = PYGGraphTransformer(
            atom_featurizer=AtomCalculator(),
            bond_featurizer=BondCalculator(self_loop=True),
            self_loop=True,
            canonical_atom_order=True,
            dtype=torch.float,
        )

    def featurize(self, smiles: Iterable[str], y: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
        """
        Featurize a list of SMILES strings
        """
        with dm.no_rdkit_log():


"""
CHDS Exposome Data Loader

Utilities for loading and processing CHDS exposome data for integration
with OpenADMET ML workflows.

The CHDS exposome includes:
- Archived pregnancy serum samples (1959-1967)
- Targeted chemical measurements (DDT, PCBs, PFAS, etc.)
- High-resolution mass spectrometry (HRMS) untargeted profiles
- Metabolomics signatures

References:
- Go et al. 2023: Exposome epidemiology for breast cancer
- Hu et al. 2020: Metabolome-wide association study of PFAS
- Hu et al. 2020: Metabolome-wide association study of DDT/DDE
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass
class CHDSChemical:
    """
    A chemical measured in CHDS archived samples.

    Attributes:
        name: Common chemical name
        smiles: SMILES structure
        cas: CAS registry number
        chds_assay: How it was measured in CHDS
        typical_range: (min, max) in ng/mL observed in CHDS
        disease_associations: Known associations from CHDS research
    """

    name: str
    smiles: str
    cas: Optional[str] = None
    chds_assay: str = "targeted"  # "targeted" or "hrms"
    typical_range: Optional[tuple[float, float]] = None
    disease_associations: Optional[list[str]] = None


# CHDS chemical database
CHDS_CHEMICAL_DATABASE = {
    # DDT and metabolites
    "p,p'-DDT": CHDSChemical(
        name="p,p'-DDT",
        smiles="Clc1ccc(cc1)C(c1ccc(Cl)cc1)C(Cl)(Cl)Cl",
        cas="50-29-3",
        chds_assay="targeted",
        typical_range=(1.0, 100.0),
        disease_associations=[
            "breast_cancer",
            "obesity",
            "hypertension",
            "testicular_cancer",
        ],
    ),
    "o,p'-DDT": CHDSChemical(
        name="o,p'-DDT",
        smiles="Clc1ccc(cc1)C(c1ccccc1Cl)C(Cl)(Cl)Cl",
        cas="789-02-6",
        chds_assay="targeted",
        typical_range=(0.1, 10.0),
        disease_associations=["her2_positive_breast_cancer"],
    ),
    "p,p'-DDE": CHDSChemical(
        name="p,p'-DDE",
        smiles="Clc1ccc(cc1)C(=C(Cl)Cl)c1ccc(Cl)cc1",
        cas="72-55-9",
        chds_assay="targeted",
        typical_range=(5.0, 200.0),
        disease_associations=["breast_cancer", "diabetes", "cognitive_impairment"],
    ),
    # PFAS
    "PFOA": CHDSChemical(
        name="Perfluorooctanoic acid",
        smiles="FC(F)(C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)O)F",
        cas="335-67-1",
        chds_assay="targeted",
        typical_range=(1.0, 50.0),
        disease_associations=["breast_cancer", "metabolic_disruption", "obesity"],
    ),
    "PFOS": CHDSChemical(
        name="Perfluorooctane sulfonate",
        smiles="FC(F)(C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)S(=O)(=O)O)F",
        cas="1763-23-1",
        chds_assay="targeted",
        typical_range=(1.0, 100.0),
        disease_associations=["breast_cancer", "thyroid_disruption"],
    ),
    # PCBs (selected congeners)
    "PCB-153": CHDSChemical(
        name="PCB-153",
        smiles="Clc1cc(Cl)c(cc1Cl)c1ccc(Cl)c(Cl)c1Cl",
        cas="35065-27-1",
        chds_assay="targeted",
        typical_range=(0.1, 5.0),
        disease_associations=["fecundity", "gestational_length", "breast_cancer"],
    ),
    "PCB-28": CHDSChemical(
        name="PCB-28",
        smiles="Clc1ccc(cc1)c1ccc(Cl)c(Cl)c1",
        cas="7012-37-5",
        chds_assay="targeted",
        disease_associations=["early_menarche", "breast_cancer"],
    ),
    "PCB-170": CHDSChemical(
        name="PCB-170",
        smiles="Clc1cc(Cl)c(c(Cl)c1Cl)c1cc(Cl)c(Cl)c(Cl)c1Cl",
        cas="35065-30-6",
        chds_assay="targeted",
        disease_associations=["early_menarche", "breast_cancer"],
    ),
    # Parathion metabolites (NEW - Cohn SOT 2026)
    "parathion": CHDSChemical(
        name="Parathion",
        smiles="CCOP(=S)(OCC)Oc1ccc(cc1)[N+](=O)[O-]",
        cas="56-38-2",
        chds_assay="hrms",
        disease_associations=["breast_cancer", "endocrine_disruption"],
    ),
    "diethylphosphoric_acid": CHDSChemical(
        name="Diethylphosphoric acid",
        smiles="CCOP(=O)(O)OCC",
        cas="598-02-7",
        chds_assay="hrms",
        disease_associations=["breast_cancer", "estrogen_disruption"],
    ),
    "4-nitrophenol": CHDSChemical(
        name="4-Nitrophenol (PNP)",
        smiles="Oc1ccc(cc1)[N+](=O)[O-]",
        cas="100-02-7",
        chds_assay="hrms",
        disease_associations=["breast_cancer", "androgen_disruption", "estrogen_disruption"],
    ),
    # Phthalates (NEW - Hu ISEE 2025 - paternal transmission)
    "diisotridecyl_phthalate": CHDSChemical(
        name="Diisotridecyl phthalate",
        smiles="CC(C)CCCCCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCCCCC(C)C",
        cas="27253-26-5",
        chds_assay="hrms",
        disease_associations=["early_menarche", "transgenerational_effects"],
    ),
    "phenoxyethanol": CHDSChemical(
        name="Phenoxyethanol",
        smiles="OCCOc1ccccc1",
        cas="122-99-6",
        chds_assay="hrms",
        disease_associations=["early_menarche", "transgenerational_effects"],
    ),
    # Additional PFAS (NEW - from posters)
    "PFHxA": CHDSChemical(
        name="Perfluorohexanoic acid",
        smiles="FC(F)(C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)O)F",
        cas="307-24-4",
        chds_assay="targeted",
        disease_associations=["early_menarche", "breast_cancer"],
    ),
}


class CHDSExposomeLoader:
    """
    Data loader for CHDS exposome data.

    Provides utilities to:
    - Load chemical measurement data
    - Prepare data for OpenADMET model training
    - Link exposures to disease outcomes
    - Generate SMILES-based feature matrices

    Example:
        >>> loader = CHDSExposomeLoader()
        >>> chemicals = loader.get_breast_cancer_chemicals()
        >>> for chem in chemicals:
        ...     print(f"{chem.name}: {chem.smiles}")
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize CHDS exposome loader.

        Args:
            data_dir: Path to CHDS data directory (if available)
        """
        self.data_dir = data_dir
        self.chemical_db = CHDS_CHEMICAL_DATABASE

    def get_all_chemicals(self) -> list[CHDSChemical]:
        """Return all chemicals in the CHDS database."""
        return list(self.chemical_db.values())

    def get_chemicals_by_disease(self, disease: str) -> list[CHDSChemical]:
        """
        Get chemicals associated with a specific disease.

        Args:
            disease: Disease name (partial match supported)

        Returns:
            List of chemicals with documented disease associations
        """
        matching = []
        for chem in self.chemical_db.values():
            if chem.disease_associations:
                for assoc in chem.disease_associations:
                    if disease.lower() in assoc.lower():
                        matching.append(chem)
                        break
        return matching

    def get_breast_cancer_chemicals(self) -> list[CHDSChemical]:
        """Get all chemicals associated with breast cancer in CHDS."""
        return self.get_chemicals_by_disease("breast_cancer")

    def get_smiles_list(
        self, chemicals: Optional[list[CHDSChemical]] = None
    ) -> list[str]:
        """
        Extract SMILES strings for a list of chemicals.

        Args:
            chemicals: List of CHDSChemical objects (None = all)

        Returns:
            List of SMILES strings
        """
        if chemicals is None:
            chemicals = self.get_all_chemicals()
        return [c.smiles for c in chemicals]

    def prepare_training_data(
        self,
        disease_endpoint: str,
        include_metabolites: bool = True,
    ) -> dict:
        """
        Prepare data for OpenADMET model training.

        Creates a dataset linking chemical structures to disease outcomes
        based on CHDS epidemiological findings.

        Args:
            disease_endpoint: Target disease for prediction
            include_metabolites: Include metabolites of parent compounds

        Returns:
            Dictionary containing:
            - 'smiles': List of SMILES strings
            - 'labels': Binary outcome labels (1 = associated)
            - 'effect_sizes': Effect sizes from CHDS studies
            - 'chemical_names': Names for reference
        """
        chemicals = self.get_chemicals_by_disease(disease_endpoint)

        result = {
            "smiles": [],
            "labels": [],
            "effect_sizes": [],
            "chemical_names": [],
            "disease_endpoint": disease_endpoint,
        }

        for chem in chemicals:
            result["smiles"].append(chem.smiles)
            result["labels"].append(1)  # Associated with disease
            result["effect_sizes"].append(None)  # To be populated from studies
            result["chemical_names"].append(chem.name)

        return result

    def load_hrms_features(
        self, feature_file: Optional[Path] = None
    ) -> Optional[dict]:
        """
        Load HRMS (untargeted exposomics) feature matrix.

        The CHDS HRMS data (Go et al. 2023) contains thousands of
        chemical features identified from archived pregnancy serum.

        Args:
            feature_file: Path to HRMS feature matrix file

        Returns:
            Dictionary with feature matrix and metadata, or None if unavailable
        """
        if feature_file is None and self.data_dir is not None:
            feature_file = self.data_dir / "hrms_features.parquet"

        if feature_file is None or not feature_file.exists():
            return None

        # Placeholder for actual loading
        return {
            "note": "HRMS feature loading requires access to CHDS data files",
            "reference": "Go et al. 2023 Environ Int",
        }

    def get_exposure_windows(self) -> dict:
        """
        Get information about exposure timing windows in CHDS.

        Returns CHDS sample collection windows and their relevance.
        """
        return {
            "2nd_trimester_serum": {
                "gestational_weeks": (13, 26),
                "samples_available": True,
                "key_findings": [
                    "IL-8 → schizophrenia (Ellman 2010)",
                    "Iron deficiency → schizophrenia (Insel 2008)",
                    "DHA deficiency → schizophrenia spectrum (Harper 2011)",
                ],
            },
            "3rd_trimester_serum": {
                "gestational_weeks": (27, 40),
                "samples_available": True,
                "key_findings": [
                    "Estrogens → breast cancer (Cohn 2017)",
                    "DDT levels → breast cancer (Cohn 2015, 2019)",
                    "PFAS → breast cancer (Cohn 2019)",
                ],
            },
            "early_postpartum_serum": {
                "gestational_weeks": None,
                "days_postpartum": (1, 3),
                "samples_available": True,
                "key_findings": [
                    "PCB exposure → breast cancer (Cohn 2012)",
                    "DDT/DDE levels → offspring fecundity (Cohn 2003)",
                ],
            },
        }

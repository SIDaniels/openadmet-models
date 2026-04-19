"""
Developmental Toxicity Prediction Module

Predicts developmental toxicity based on gestational timing windows,
informed by CHDS findings on critical periods of susceptibility.

Key windows modeled (from CHDS research):
- 1st trimester: Organogenesis, neural tube formation
- 2nd trimester: Brain development (IL-8/CXCR2 pathway, DHA requirements)
- 3rd trimester: Endocrine programming (estrogens, DDT effects)
- Perinatal: Oxytocin receptor programming, birth complications

References:
- Ellman 2010: Maternal IL-8 → schizophrenia brain structure
- Insel 2008: Maternal iron deficiency → schizophrenia
- Freedman 2015: Perinatal oxytocin → bipolar disorder
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class GestationalWindow(Enum):
    """Gestational timing windows with distinct toxicity profiles."""

    PRECONCEPTION = "preconception"
    FIRST_TRIMESTER = "trimester_1"  # Weeks 1-12
    SECOND_TRIMESTER = "trimester_2"  # Weeks 13-26
    THIRD_TRIMESTER = "trimester_3"  # Weeks 27-40
    PERINATAL = "perinatal"  # Birth ± 2 weeks
    POSTNATAL_EARLY = "postnatal_early"  # 0-6 months
    POSTNATAL_LATE = "postnatal_late"  # 6-24 months


@dataclass
class DevelopmentalToxicityEndpoint:
    """
    A developmental toxicity endpoint specification.

    Attributes:
        name: Endpoint name (e.g., "neural_tube_defects", "low_birth_weight")
        sensitive_windows: Which gestational windows are most sensitive
        target_pathways: Molecular pathways involved
        chds_evidence: Key CHDS studies supporting this endpoint
    """

    name: str
    sensitive_windows: list[GestationalWindow]
    target_pathways: list[str]
    chds_evidence: list[str] = field(default_factory=list)


# Pre-defined endpoints based on CHDS research
CHDS_DEVELOPMENTAL_ENDPOINTS = {
    "breast_cancer_risk": DevelopmentalToxicityEndpoint(
        name="breast_cancer_risk",
        sensitive_windows=[
            GestationalWindow.FIRST_TRIMESTER,
            GestationalWindow.THIRD_TRIMESTER,
        ],
        target_pathways=["HER2", "ERα", "mammary_gland_programming"],
        chds_evidence=[
            "Cohn 2015 JCEM: o,p'-DDT → HER2+ breast cancer",
            "Cohn 2019 JNCI: DDT induction time and susceptibility windows",
            "Cohn 2017 JCEM: 3rd trimester estrogens → breast cancer",
        ],
    ),
    "schizophrenia_risk": DevelopmentalToxicityEndpoint(
        name="schizophrenia_risk",
        sensitive_windows=[GestationalWindow.SECOND_TRIMESTER],
        target_pathways=["CXCR1", "CXCR2", "DHA_resolution", "iron_metabolism"],
        chds_evidence=[
            "Ellman 2010: Maternal IL-8 → schizophrenia brain structure",
            "Insel 2008: Maternal iron deficiency → schizophrenia",
            "Harper 2011: Low maternal DHA → schizophrenia spectrum",
        ],
    ),
    "obesity_programming": DevelopmentalToxicityEndpoint(
        name="obesity_programming",
        sensitive_windows=[
            GestationalWindow.FIRST_TRIMESTER,
            GestationalWindow.SECOND_TRIMESTER,
        ],
        target_pathways=["PPARα", "PPARγ", "adipogenesis", "lipid_metabolism"],
        chds_evidence=[
            "La Merrill 2020: DDT exposure → obesity in middle age",
            "Cirillo 2021: Grandmaternal DDT → granddaughter obesity",
            "Cohn 2020: PFAS → metabolic disruption across generations",
        ],
    ),
    "cardiovascular_programming": DevelopmentalToxicityEndpoint(
        name="cardiovascular_programming",
        sensitive_windows=[
            GestationalWindow.THIRD_TRIMESTER,
            GestationalWindow.PERINATAL,
        ],
        target_pathways=["sFlt1_VEGF", "endothelin1", "NF_kB", "vascular_remodeling"],
        chds_evidence=[
            "Cirillo 2015 Circulation: Pregnancy complications → CVD death",
            "Mongraw-Chaffin 2010: Preeclampsia → CVD mortality",
        ],
    ),
    "neurodevelopmental_impairment": DevelopmentalToxicityEndpoint(
        name="neurodevelopmental_impairment",
        sensitive_windows=[
            GestationalWindow.SECOND_TRIMESTER,
            GestationalWindow.PERINATAL,
        ],
        target_pathways=["OXTR", "dopamine_synthesis", "neuronal_migration"],
        chds_evidence=[
            "Freedman 2015: Perinatal oxytocin → bipolar disorder",
            "Pike 2024: Prenatal inflammation → adolescent depression",
            "Mohyee 2025: Prenatal inflammation → hippocampal neurite density",
        ],
    ),
    "colorectal_cancer_risk": DevelopmentalToxicityEndpoint(
        name="colorectal_cancer_risk",
        sensitive_windows=[
            GestationalWindow.FIRST_TRIMESTER,
            GestationalWindow.POSTNATAL_EARLY,
        ],
        target_pathways=["gut_microbiome", "antibiotic_effects", "immune_programming"],
        chds_evidence=[
            "Murphy 2023 IJE: In utero antibiotics → colorectal cancer",
            "Murphy 2022 Gut: Maternal obesity → colorectal cancer",
            "Murphy 2024: Childhood antibiotics → adult colorectal cancer",
        ],
    ),
}


class DevelopmentalToxicity:
    """
    Developmental toxicity predictor incorporating gestational windows.

    This class wraps OpenADMET models to provide developmental toxicity
    predictions that account for timing of exposure during gestation.

    Example:
        >>> dev_tox = DevelopmentalToxicity()
        >>> # Predict breast cancer risk from in utero DDT exposure
        >>> result = dev_tox.predict(
        ...     smiles="ClC(Cl)=C(c1ccc(Cl)cc1)c1ccc(Cl)cc1",  # DDT
        ...     window=GestationalWindow.FIRST_TRIMESTER,
        ...     endpoint="breast_cancer_risk"
        ... )
    """

    def __init__(self, model_type: str = "chemprop"):
        """
        Initialize developmental toxicity predictor.

        Args:
            model_type: Base model architecture from OpenADMET
                       ("chemprop", "rf", "xgboost", etc.)
        """
        self.model_type = model_type
        self.endpoints = CHDS_DEVELOPMENTAL_ENDPOINTS
        self._models: dict = {}

    def available_endpoints(self) -> list[str]:
        """Return list of available developmental toxicity endpoints."""
        return list(self.endpoints.keys())

    def get_endpoint_info(self, endpoint: str) -> DevelopmentalToxicityEndpoint:
        """Get detailed information about a specific endpoint."""
        if endpoint not in self.endpoints:
            raise ValueError(
                f"Unknown endpoint: {endpoint}. "
                f"Available: {self.available_endpoints()}"
            )
        return self.endpoints[endpoint]

    def predict(
        self,
        smiles: str | list[str],
        window: GestationalWindow,
        endpoint: str,
        return_uncertainty: bool = True,
    ) -> dict:
        """
        Predict developmental toxicity for a compound.

        Args:
            smiles: SMILES string(s) for compound(s) to predict
            window: Gestational window of exposure
            endpoint: Which developmental endpoint to predict
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Dictionary containing:
            - 'prediction': Toxicity score (0-1)
            - 'window_sensitivity': How sensitive this window is for this endpoint
            - 'uncertainty': Prediction uncertainty (if requested)
            - 'relevant_pathways': Target pathways involved
        """
        endpoint_info = self.get_endpoint_info(endpoint)

        # Check window sensitivity
        window_sensitive = window in endpoint_info.sensitive_windows
        sensitivity_modifier = 1.5 if window_sensitive else 0.5

        # Placeholder for actual model prediction
        # In production, this would call the underlying OpenADMET model
        if isinstance(smiles, str):
            smiles = [smiles]

        result = {
            "smiles": smiles,
            "endpoint": endpoint,
            "window": window.value,
            "window_sensitive": window_sensitive,
            "sensitivity_modifier": sensitivity_modifier,
            "relevant_pathways": endpoint_info.target_pathways,
            "chds_evidence": endpoint_info.chds_evidence,
            "prediction": None,  # To be filled by actual model
            "uncertainty": None if not return_uncertainty else None,
        }

        return result

    def predict_all_windows(
        self,
        smiles: str,
        endpoint: str,
    ) -> dict[GestationalWindow, dict]:
        """
        Predict toxicity across all gestational windows.

        Useful for identifying critical windows of susceptibility.

        Args:
            smiles: SMILES string for compound
            endpoint: Developmental endpoint to predict

        Returns:
            Dictionary mapping each window to its prediction results
        """
        results = {}
        for window in GestationalWindow:
            results[window] = self.predict(smiles, window, endpoint)
        return results

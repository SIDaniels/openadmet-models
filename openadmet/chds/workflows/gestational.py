"""
Gestational Biomarker Model

Workflow for predicting disease outcomes from gestational biomarkers,
based on CHDS's unique prospective pregnancy serum data.

Key biomarker → disease relationships from CHDS:
- 3rd trimester estrogens → breast cancer (Cohn 2017)
- 2nd trimester IL-8 → schizophrenia (Ellman 2010)
- 2nd trimester iron → schizophrenia (Insel 2008)
- 2nd trimester DHA → schizophrenia spectrum (Harper 2011)
- DDT levels → HER2+ breast cancer (Cohn 2015)
- Amino acid profiles → breast cancer (Teeny 2023)

This represents the only prospective human data showing that
the hormonal/metabolic milieu of a specific pregnancy trimester
encodes long-term disease risk in offspring.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class BiomarkerType(Enum):
    """Types of gestational biomarkers measured in CHDS."""

    HORMONE = "hormone"
    CYTOKINE = "cytokine"
    NUTRIENT = "nutrient"
    XENOBIOTIC = "xenobiotic"
    METABOLITE = "metabolite"
    PROTEIN = "protein"


@dataclass
class GestationalBiomarker:
    """
    A gestational biomarker measured in CHDS.

    Attributes:
        name: Biomarker name
        type: Category of biomarker
        measurement_trimester: When measured in pregnancy
        disease_association: Disease it predicts
        direction: "high" or "low" associated with risk
        effect_size: Hazard ratio or odds ratio
        chds_study: Key publication
    """

    name: str
    type: BiomarkerType
    measurement_trimester: int  # 1, 2, or 3
    disease_association: str
    direction: str  # "high" or "low"
    effect_size: Optional[float] = None
    chds_study: Optional[str] = None


# CHDS validated gestational biomarkers
CHDS_GESTATIONAL_BIOMARKERS = {
    "estrone_3rd": GestationalBiomarker(
        name="Estrone (E1)",
        type=BiomarkerType.HORMONE,
        measurement_trimester=3,
        disease_association="breast_cancer",
        direction="high",
        effect_size=1.5,
        chds_study="Cohn 2017 JCEM",
    ),
    "estradiol_3rd": GestationalBiomarker(
        name="Estradiol (E2)",
        type=BiomarkerType.HORMONE,
        measurement_trimester=3,
        disease_association="breast_cancer",
        direction="high",
        effect_size=1.4,
        chds_study="Cohn 2017 JCEM",
    ),
    "estriol_3rd": GestationalBiomarker(
        name="Estriol (E3)",
        type=BiomarkerType.HORMONE,
        measurement_trimester=3,
        disease_association="breast_cancer",
        direction="high",
        effect_size=1.3,
        chds_study="Cohn 2017 JCEM",
    ),
    "il8_2nd": GestationalBiomarker(
        name="Interleukin-8 (IL-8/CXCL8)",
        type=BiomarkerType.CYTOKINE,
        measurement_trimester=2,
        disease_association="schizophrenia",
        direction="high",
        effect_size=2.0,
        chds_study="Ellman 2010 Schizophr Res",
    ),
    "ferritin_2nd": GestationalBiomarker(
        name="Ferritin (iron status)",
        type=BiomarkerType.NUTRIENT,
        measurement_trimester=2,
        disease_association="schizophrenia",
        direction="low",
        effect_size=1.8,
        chds_study="Insel 2008 Arch Gen Psychiatry",
    ),
    "dha_2nd": GestationalBiomarker(
        name="DHA (docosahexaenoic acid)",
        type=BiomarkerType.NUTRIENT,
        measurement_trimester=2,
        disease_association="schizophrenia_spectrum",
        direction="low",
        effect_size=1.6,
        chds_study="Harper 2011 Schizophr Res",
    ),
    "ddt_3rd": GestationalBiomarker(
        name="DDT (dichlorodiphenyltrichloroethane)",
        type=BiomarkerType.XENOBIOTIC,
        measurement_trimester=3,
        disease_association="breast_cancer",
        direction="high",
        effect_size=3.7,
        chds_study="Cohn 2015 JCEM; Cohn 2019 JNCI",
    ),
    "pfas_2nd": GestationalBiomarker(
        name="PFAS (long-chain)",
        type=BiomarkerType.XENOBIOTIC,
        measurement_trimester=2,
        disease_association="breast_cancer",
        direction="high",
        effect_size=1.5,
        chds_study="Cohn 2019 Reprod Toxicol",
    ),
    "amino_acids_3rd": GestationalBiomarker(
        name="Amino acid profile",
        type=BiomarkerType.METABOLITE,
        measurement_trimester=3,
        disease_association="early_breast_cancer",
        direction="altered",
        chds_study="Teeny 2023 Research Square; Teeny 2025 Reprod Toxicol",
    ),
}


class GestationalBiomarkerModel:
    """
    Model for predicting disease from gestational biomarkers.

    This workflow integrates CHDS's unique prospective biomarker data
    with OpenADMET predictions to:
    1. Identify individuals at elevated disease risk
    2. Suggest monitoring and intervention strategies
    3. Link biomarker patterns to druggable pathways

    Example:
        >>> model = GestationalBiomarkerModel()
        >>> risk = model.predict_disease_risk(
        ...     biomarker_values={
        ...         "estradiol_3rd": 150.0,  # ng/mL
        ...         "ddt_3rd": 25.0,  # ng/mL
        ...     },
        ...     disease="breast_cancer"
        ... )
    """

    def __init__(self):
        """Initialize gestational biomarker model."""
        self.biomarkers = CHDS_GESTATIONAL_BIOMARKERS

    def get_biomarkers_for_disease(self, disease: str) -> list[GestationalBiomarker]:
        """Get all biomarkers associated with a disease."""
        matching = []
        for bm in self.biomarkers.values():
            if disease.lower() in bm.disease_association.lower():
                matching.append(bm)
        return matching

    def predict_disease_risk(
        self,
        biomarker_values: dict[str, float],
        disease: str,
        return_interventions: bool = True,
    ) -> dict:
        """
        Predict disease risk from gestational biomarker values.

        Args:
            biomarker_values: Dict mapping biomarker names to values
            disease: Target disease
            return_interventions: Include intervention suggestions

        Returns:
            Dictionary containing:
            - 'disease': Target disease
            - 'risk_score': Composite risk score
            - 'contributing_biomarkers': Which biomarkers contribute
            - 'interventions': Suggested interventions (if requested)
        """
        relevant_bms = self.get_biomarkers_for_disease(disease)

        contributing = []
        risk_factors = []

        for bm in relevant_bms:
            bm_key = None
            for key in biomarker_values:
                if bm.name.lower() in key.lower() or key.lower() in bm.name.lower():
                    bm_key = key
                    break

            if bm_key:
                value = biomarker_values[bm_key]
                # Placeholder risk calculation
                if bm.effect_size:
                    risk_factors.append(bm.effect_size)
                contributing.append(
                    {
                        "biomarker": bm.name,
                        "value": value,
                        "direction": bm.direction,
                        "effect_size": bm.effect_size,
                        "study": bm.chds_study,
                    }
                )

        # Compute composite risk
        if risk_factors:
            composite_risk = np.prod(risk_factors) ** (1 / len(risk_factors))
        else:
            composite_risk = 1.0

        result = {
            "disease": disease,
            "risk_score": composite_risk,
            "risk_category": self._categorize_risk(composite_risk),
            "contributing_biomarkers": contributing,
            "n_biomarkers_measured": len(biomarker_values),
            "n_biomarkers_relevant": len(contributing),
        }

        if return_interventions:
            result["interventions"] = self._suggest_interventions(
                disease, contributing
            )

        return result

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into levels."""
        if risk_score >= 2.0:
            return "high"
        elif risk_score >= 1.5:
            return "elevated"
        elif risk_score >= 1.2:
            return "slightly_elevated"
        else:
            return "baseline"

    def _suggest_interventions(
        self, disease: str, contributing: list[dict]
    ) -> list[dict]:
        """Suggest interventions based on biomarker profile."""
        interventions = []

        # Disease-specific interventions based on CHDS findings
        if "breast_cancer" in disease.lower():
            if any("ddt" in c["biomarker"].lower() for c in contributing):
                interventions.append(
                    {
                        "type": "screening",
                        "recommendation": (
                            "Enhanced HER2 screening given DDT exposure. "
                            "Consider earlier mammography initiation."
                        ),
                        "rationale": "CHDS: DDT exposure → HER2+ breast cancer",
                        "study": "Cohn 2015 JCEM",
                    }
                )
            if any("estro" in c["biomarker"].lower() for c in contributing):
                interventions.append(
                    {
                        "type": "chemoprevention",
                        "recommendation": (
                            "Consider ERα-targeted chemoprevention "
                            "(aromatase inhibitors, SERMs) at earlier age."
                        ),
                        "rationale": "CHDS: Elevated pregnancy estrogens → breast cancer",
                        "study": "Cohn 2017 JCEM",
                    }
                )

        if "schizophrenia" in disease.lower():
            if any("il-8" in c["biomarker"].lower() or "il8" in c["biomarker"].lower()
                   for c in contributing):
                interventions.append(
                    {
                        "type": "monitoring",
                        "recommendation": (
                            "Enhanced neurodevelopmental monitoring. "
                            "CXCR2 antagonists are in development."
                        ),
                        "rationale": "CHDS: Maternal IL-8 → reduced gray matter",
                        "study": "Ellman 2010 Schizophr Res",
                    }
                )
            if any("iron" in c["biomarker"].lower() or "ferritin" in c["biomarker"].lower()
                   for c in contributing):
                interventions.append(
                    {
                        "type": "supplementation",
                        "recommendation": (
                            "Ensure adequate iron supplementation in pregnancy. "
                            "This is already standard prenatal care."
                        ),
                        "rationale": "CHDS: Iron deficiency → schizophrenia",
                        "study": "Insel 2008 Arch Gen Psychiatry",
                    }
                )

        return interventions

    def get_prevention_window(self, disease: str) -> dict:
        """
        Identify the optimal prevention window for a disease.

        Based on CHDS findings about which gestational windows matter.
        """
        bms = self.get_biomarkers_for_disease(disease)

        windows = {}
        for bm in bms:
            trimester = bm.measurement_trimester
            if trimester not in windows:
                windows[trimester] = []
            windows[trimester].append(bm.name)

        # Identify critical window
        if windows:
            critical = max(windows.keys(), key=lambda x: len(windows[x]))
        else:
            critical = None

        return {
            "disease": disease,
            "windows_by_trimester": windows,
            "critical_window": f"trimester_{critical}" if critical else None,
            "intervention_timing": (
                f"Interventions targeting {disease} should focus on "
                f"trimester {critical} based on CHDS biomarker data."
                if critical else "No specific window identified."
            ),
        }

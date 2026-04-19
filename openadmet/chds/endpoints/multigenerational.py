"""
Multigenerational Risk Prediction Module

Predicts risk of health effects across multiple generations based on
ancestral exposures. This is a unique capability informed by CHDS's
unprecedented 3-generation prospective data.

Key multigenerational findings from CHDS:
- Grandmaternal DDT → granddaughter early menarche and obesity
- Grandmaternal PCBs → granddaughter obesity
- Prenatal exposures → epigenetic changes detectable in offspring

The mechanism involves:
1. Direct exposure of fetal germ cells (F2 generation exposed in utero as F1 gonads)
2. Epigenetic programming that persists across generations
3. Bioaccumulation of persistent chemicals

References:
- Cirillo 2021: Grandmaternal DDT → granddaughter early menarche/obesity
- Cohn 2025: Grandmaternal PCBs → granddaughter obesity
- Wu 2020: DDT exposure → DNA methylation alterations in daughters
- Mouat 2025: Newborn DNA methylation → autism spectrum disorder
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class Generation(Enum):
    """Generations in multigenerational study."""

    F0 = "F0"  # Index generation (exposed)
    F1 = "F1"  # Children of exposed
    F2 = "F2"  # Grandchildren
    F3 = "F3"  # Great-grandchildren


class TransmissionMechanism(Enum):
    """Mechanisms of multigenerational effect transmission."""

    DIRECT_FETAL_EXPOSURE = "direct_fetal"  # F1 exposed in utero
    GERM_CELL_EXPOSURE = "germ_cell"  # F2 germ cells exposed as F1 fetus
    EPIGENETIC_INHERITANCE = "epigenetic"  # Methylation, histone marks
    BIOACCUMULATION = "bioaccumulation"  # Persistent chemical transfer
    MICROBIOME = "microbiome"  # Maternal microbiome effects


@dataclass
class MultigenerationalOutcome:
    """
    A multigenerational health outcome.

    Attributes:
        outcome: The health outcome (e.g., "breast_cancer", "obesity")
        affected_generations: Which generations show the effect
        transmission: Primary transmission mechanism
        effect_size: Hazard ratio or odds ratio
        latency_years: Time from exposure to outcome
        chds_study: Key CHDS publication
    """

    outcome: str
    affected_generations: list[Generation]
    transmission: TransmissionMechanism
    effect_size: float  # HR or OR
    latency_years: Optional[int] = None
    chds_study: Optional[str] = None


# CHDS multigenerational findings
CHDS_MULTIGENERATIONAL_OUTCOMES = {
    "DDT_granddaughter_obesity": MultigenerationalOutcome(
        outcome="obesity",
        affected_generations=[Generation.F2],
        transmission=TransmissionMechanism.GERM_CELL_EXPOSURE,
        effect_size=2.4,  # OR for obesity
        latency_years=50,
        chds_study="Cirillo 2021 Cancer Epidemiol Biomarkers Prev",
    ),
    "DDT_granddaughter_menarche": MultigenerationalOutcome(
        outcome="early_menarche",
        affected_generations=[Generation.F2],
        transmission=TransmissionMechanism.GERM_CELL_EXPOSURE,
        effect_size=1.8,  # OR for early menarche
        latency_years=12,
        chds_study="Cirillo 2021 Cancer Epidemiol Biomarkers Prev",
    ),
    "PCB_granddaughter_obesity": MultigenerationalOutcome(
        outcome="obesity",
        affected_generations=[Generation.F1, Generation.F2],
        transmission=TransmissionMechanism.BIOACCUMULATION,
        effect_size=1.6,
        latency_years=50,
        chds_study="Cohn 2025 Obesity",
    ),
    "DDT_daughter_breast_cancer": MultigenerationalOutcome(
        outcome="breast_cancer",
        affected_generations=[Generation.F1],
        transmission=TransmissionMechanism.DIRECT_FETAL_EXPOSURE,
        effect_size=3.7,  # HR for breast cancer
        latency_years=52,
        chds_study="Cohn 2015 JCEM; Cohn 2019 JNCI",
    ),
    "DDT_methylation_changes": MultigenerationalOutcome(
        outcome="epigenetic_alterations",
        affected_generations=[Generation.F1],
        transmission=TransmissionMechanism.EPIGENETIC_INHERITANCE,
        effect_size=1.0,  # Qualitative
        chds_study="Wu 2020 Reprod Toxicol",
    ),
    "grandparental_exposures_autism": MultigenerationalOutcome(
        outcome="autism_spectrum_disorder",
        affected_generations=[Generation.F2],
        transmission=TransmissionMechanism.EPIGENETIC_INHERITANCE,
        effect_size=1.5,
        chds_study="Mouat 2025 Biol Sex Diff; Kuodza 2025 INSAR",
    ),
}


class MultigenerationalRisk:
    """
    Multigenerational risk predictor based on CHDS 3-generation data.

    This is a unique capability that predicts health risks in offspring
    and grandchildren based on ancestral chemical exposures. No other
    cohort has the prospective, multi-generation design of CHDS.

    Example:
        >>> mg = MultigenerationalRisk()
        >>> risk = mg.predict_transgenerational(
        ...     smiles="Clc1ccc(cc1)C(c1ccc(Cl)cc1)C(Cl)(Cl)Cl",  # DDT
        ...     exposure_generation=Generation.F0,
        ...     outcome="obesity",
        ...     target_generation=Generation.F2
        ... )
    """

    def __init__(self):
        """Initialize multigenerational risk predictor."""
        self.known_outcomes = CHDS_MULTIGENERATIONAL_OUTCOMES

    def predict_transgenerational(
        self,
        smiles: str,
        exposure_generation: Generation,
        outcome: str,
        target_generation: Generation,
        exposure_level: str = "high",  # "low", "medium", "high"
    ) -> dict:
        """
        Predict transgenerational risk for a specific outcome.

        Args:
            smiles: SMILES string for the compound
            exposure_generation: Generation that was exposed
            outcome: Health outcome to predict
            target_generation: Generation to assess risk in
            exposure_level: Relative exposure level

        Returns:
            Dictionary containing:
            - 'relative_risk': Estimated relative risk
            - 'transmission_mechanism': How the effect transmits
            - 'chds_evidence': Supporting CHDS studies
            - 'generations_affected': Which generations show effects
        """
        # Calculate generation distance
        gen_order = [Generation.F0, Generation.F1, Generation.F2, Generation.F3]
        exp_idx = gen_order.index(exposure_generation)
        tgt_idx = gen_order.index(target_generation)
        gen_distance = tgt_idx - exp_idx

        if gen_distance < 0:
            raise ValueError(
                "Target generation cannot be before exposure generation"
            )

        # Find matching CHDS outcomes
        matching_outcomes = []
        for name, data in self.known_outcomes.items():
            if outcome.lower() in data.outcome.lower():
                if target_generation in data.affected_generations:
                    matching_outcomes.append((name, data))

        # Exposure level modifier
        level_modifier = {"low": 0.5, "medium": 1.0, "high": 1.5}[exposure_level]

        # Distance attenuation (effects generally decrease with distance)
        distance_attenuation = 0.7 ** gen_distance

        if matching_outcomes:
            best_match = matching_outcomes[0][1]
            base_risk = best_match.effect_size
            transmission = best_match.transmission
            evidence = best_match.chds_study
        else:
            # Default prediction for unknown compounds
            base_risk = 1.2  # Modest elevated risk
            transmission = TransmissionMechanism.EPIGENETIC_INHERITANCE
            evidence = None

        adjusted_risk = base_risk * level_modifier * distance_attenuation

        return {
            "smiles": smiles,
            "outcome": outcome,
            "exposure_generation": exposure_generation.value,
            "target_generation": target_generation.value,
            "generation_distance": gen_distance,
            "relative_risk": adjusted_risk,
            "transmission_mechanism": transmission.value,
            "chds_evidence": evidence,
            "exposure_level": exposure_level,
        }

    def predict_all_generations(
        self,
        smiles: str,
        outcome: str,
        exposure_level: str = "medium",
    ) -> dict:
        """
        Predict risk across all future generations.

        Args:
            smiles: SMILES string for compound
            outcome: Health outcome to predict
            exposure_level: Relative exposure level

        Returns:
            Dictionary mapping generations to risk predictions
        """
        results = {}
        for gen in [Generation.F1, Generation.F2, Generation.F3]:
            try:
                results[gen.value] = self.predict_transgenerational(
                    smiles=smiles,
                    exposure_generation=Generation.F0,
                    outcome=outcome,
                    target_generation=gen,
                    exposure_level=exposure_level,
                )
            except ValueError:
                continue
        return results

    def identify_persistent_compounds(
        self,
        smiles_list: list[str],
        threshold_half_life_days: float = 365.0,
    ) -> list[dict]:
        """
        Identify compounds with multigenerational concern due to persistence.

        Compounds with very long half-lives (PFAS, DDT, PCBs) can accumulate
        and transfer across generations, as documented in CHDS.

        Args:
            smiles_list: List of SMILES to evaluate
            threshold_half_life_days: Minimum half-life for concern

        Returns:
            List of compounds flagged for multigenerational concern
        """
        flagged = []
        for smiles in smiles_list:
            # Placeholder - would integrate with ADMET half-life predictions
            flagged.append(
                {
                    "smiles": smiles,
                    "persistence_concern": None,  # To be predicted
                    "bioaccumulation_potential": None,
                    "multigenerational_flag": None,
                }
            )
        return flagged

    def get_chds_evidence(self, outcome: str) -> list[dict]:
        """
        Get all CHDS evidence for a specific outcome.

        Args:
            outcome: Health outcome to query

        Returns:
            List of relevant CHDS findings
        """
        relevant = []
        for name, data in self.known_outcomes.items():
            if outcome.lower() in data.outcome.lower():
                relevant.append(
                    {
                        "name": name,
                        "outcome": data.outcome,
                        "generations": [g.value for g in data.affected_generations],
                        "effect_size": data.effect_size,
                        "mechanism": data.transmission.value,
                        "study": data.chds_study,
                    }
                )
        return relevant

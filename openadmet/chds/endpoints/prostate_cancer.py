"""
Prostate Cancer Metabolomics Endpoint

CHDS findings on race-specific metabolic pathways predicting lethal prostate cancer.
Based on Krigbaum et al. SOT 2026 poster findings.

Key findings:
- Black men: Oxidative stress pathways (selenocysteine, acetoacetate)
- Non-Black men: Inflammatory pathways (arachidonic acid, prostaglandins, leukotrienes)
- Pre-diagnostic serum at mean age 34 predicts lethal cancer decades later
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ProstateCancerPathway(Enum):
    """Metabolic pathway categories from CHDS prostate cancer MWAS."""

    OXIDATIVE_STRESS = "oxidative_stress"
    INFLAMMATION = "inflammation"
    LIPID_PEROXIDATION = "lipid_peroxidation"
    AMINO_ACID = "amino_acid"
    VITAMIN = "vitamin"


@dataclass
class ProstateCancerBiomarker:
    """
    A biomarker associated with lethal prostate cancer in CHDS.

    Attributes:
        name: Biomarker name
        pathway: Metabolic pathway category
        race_specific: Which race group shows this association
        direction: "up" or "down" in lethal cases
        chds_study: Reference
    """

    name: str
    pathway: ProstateCancerPathway
    race_specific: str  # "black", "non_black", or "both"
    direction: str  # "up" or "down" in cases
    chds_study: str = "Krigbaum SOT 2026"


# CHDS prostate cancer biomarkers from MWAS
CHDS_PROSTATE_BIOMARKERS = {
    # Black men - oxidative stress
    "selenocysteine": ProstateCancerBiomarker(
        name="Selenocysteine",
        pathway=ProstateCancerPathway.OXIDATIVE_STRESS,
        race_specific="black",
        direction="down",
    ),
    "acetoacetate": ProstateCancerBiomarker(
        name="Acetoacetate",
        pathway=ProstateCancerPathway.OXIDATIVE_STRESS,
        race_specific="black",
        direction="up",
    ),
    "linolenic_acid_black": ProstateCancerBiomarker(
        name="Linolenic acid",
        pathway=ProstateCancerPathway.LIPID_PEROXIDATION,
        race_specific="black",
        direction="down",
    ),
    # Non-Black men - inflammatory
    "arachidonic_acid": ProstateCancerBiomarker(
        name="Arachidonic acid",
        pathway=ProstateCancerPathway.INFLAMMATION,
        race_specific="non_black",
        direction="up",
    ),
    "15_hete": ProstateCancerBiomarker(
        name="15-Hydroxyeicosatetraenoic acid (15-HETE)",
        pathway=ProstateCancerPathway.INFLAMMATION,
        race_specific="non_black",
        direction="up",
    ),
    "leukotriene_a4": ProstateCancerBiomarker(
        name="Leukotriene A4",
        pathway=ProstateCancerPathway.INFLAMMATION,
        race_specific="non_black",
        direction="up",
    ),
    "prostaglandin_pge2": ProstateCancerBiomarker(
        name="Prostaglandin E2/D2",
        pathway=ProstateCancerPathway.INFLAMMATION,
        race_specific="non_black",
        direction="up",
    ),
    "prostaglandin_pgf": ProstateCancerBiomarker(
        name="Prostaglandin F-type",
        pathway=ProstateCancerPathway.INFLAMMATION,
        race_specific="non_black",
        direction="up",
    ),
    "14_15_dhet": ProstateCancerBiomarker(
        name="14,15-DHET",
        pathway=ProstateCancerPathway.INFLAMMATION,
        race_specific="non_black",
        direction="up",
    ),
    "linolenic_acid_nonblack": ProstateCancerBiomarker(
        name="Linolenic acid",
        pathway=ProstateCancerPathway.LIPID_PEROXIDATION,
        race_specific="non_black",
        direction="up",
    ),
}

# Pathway enrichment by race from MWAS
PATHWAY_ENRICHMENT = {
    "black": {
        "oxidative_stress": [
            "Selenoamino acid metabolism",
            "Methionine and cysteine metabolism",
            "Glutathione metabolism",
            "Pentose phosphate pathway",
            "Vitamin E metabolism",
        ],
        "inflammation": [
            "Arachidonic acid metabolism",  # less significant
        ],
    },
    "non_black": {
        "inflammation": [
            "Arachidonic acid metabolism",
            "Leukotriene metabolism",
            "Prostaglandin formation from arachidonate",
            "Linoleate metabolism",
            "Glycerophospholipid metabolism",
        ],
        "oxidative_stress": [
            "Squalene and cholesterol biosynthesis",
            "Glutathione metabolism",
        ],
    },
}


class ProstateCancerRisk:
    """
    Model for predicting lethal prostate cancer risk.

    Uses race-stratified metabolic signatures from CHDS MWAS.

    Example:
        >>> model = ProstateCancerRisk()
        >>> risk = model.assess_risk(
        ...     race="black",
        ...     selenocysteine_level="low",
        ...     acetoacetate_level="high"
        ... )
    """

    def __init__(self):
        """Initialize prostate cancer risk model."""
        self.biomarkers = CHDS_PROSTATE_BIOMARKERS
        self.pathways = PATHWAY_ENRICHMENT

    def get_biomarkers_by_race(self, race: str) -> list[ProstateCancerBiomarker]:
        """Get biomarkers relevant to a specific race group."""
        return [
            bm for bm in self.biomarkers.values()
            if bm.race_specific == race or bm.race_specific == "both"
        ]

    def get_primary_pathways(self, race: str) -> list[str]:
        """Get primary metabolic pathways for a race group."""
        if race == "black":
            return ["oxidative_stress"]
        elif race == "non_black":
            return ["inflammation"]
        return ["oxidative_stress", "inflammation"]

    def assess_risk(
        self,
        race: str,
        biomarker_values: Optional[dict[str, str]] = None,
    ) -> dict:
        """
        Assess lethal prostate cancer risk based on metabolic profile.

        Args:
            race: "black" or "non_black"
            biomarker_values: Dict of biomarker names to "high", "low", or "normal"

        Returns:
            Risk assessment with pathway contributions
        """
        relevant_biomarkers = self.get_biomarkers_by_race(race)
        primary_pathways = self.get_primary_pathways(race)

        risk_factors = []
        if biomarker_values:
            for bm in relevant_biomarkers:
                bm_key = bm.name.lower().replace(" ", "_").replace("-", "_")
                if bm_key in biomarker_values:
                    value = biomarker_values[bm_key]
                    # Risk increases when value matches disease direction
                    if (bm.direction == "up" and value == "high") or \
                       (bm.direction == "down" and value == "low"):
                        risk_factors.append({
                            "biomarker": bm.name,
                            "value": value,
                            "expected_direction": bm.direction,
                            "contributes_to_risk": True,
                        })

        return {
            "race": race,
            "primary_pathways": primary_pathways,
            "n_risk_factors": len(risk_factors),
            "risk_factors": risk_factors,
            "recommendations": self._get_recommendations(race, risk_factors),
        }

    def _get_recommendations(
        self, race: str, risk_factors: list[dict]
    ) -> list[str]:
        """Generate recommendations based on risk profile."""
        recommendations = []

        if race == "black" and any(
            rf["biomarker"] == "Selenocysteine" for rf in risk_factors
        ):
            recommendations.append(
                "Consider selenium supplementation - selenocysteine deficiency "
                "linked to accelerated prostate carcinogenesis in mouse models."
            )

        if race == "non_black" and len(risk_factors) > 0:
            recommendations.append(
                "Anti-inflammatory interventions may be beneficial - "
                "arachidonic acid pathway activation linked to lethal prostate cancer."
            )

        if not recommendations:
            recommendations.append(
                "Standard prostate cancer screening per guidelines."
            )

        return recommendations

    def get_study_info(self) -> dict:
        """Return information about the CHDS prostate cancer study."""
        return {
            "study": "Child Health and Development Studies",
            "design": "Prospective 60-year follow-up",
            "sample_timing": "Pre-diagnostic serum at mean age 34",
            "cases_black": 111,
            "controls_black": 220,
            "cases_non_black": 258,
            "controls_non_black": 515,
            "outcome": "Lethal prostate cancer",
            "methods": "MWAS using Rodin/Mummichog",
            "reference": "Krigbaum et al. SOT 2026 Abstract #5132",
        }

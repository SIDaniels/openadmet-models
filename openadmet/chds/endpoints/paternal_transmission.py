"""
Paternal and Grandparental Transmission of Health Risks

CHDS findings on male-line transmission of metabolic and reproductive risks.
Based on Hu et al. ISEE 2025 poster - three-generation metabolomics.

Key findings:
- Grandpaternal metabolism strongly affects F2 (granddaughter) outcomes
- Paternal effects STRONGER in F2 than F1 (opposite for maternal)
- Grandparental metabolites show consistent antagonism
- Environmental chemicals in grandfather serum linked to earlier menarche
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TransmissionLine(Enum):
    """Parental line of transmission."""

    MATERNAL = "maternal"
    PATERNAL = "paternal"
    BOTH = "both"


class MetabolicPathway(Enum):
    """Metabolic pathway categories from three-generation MWAS."""

    AMINO_ACID = "amino_acid_metabolism"
    LIPID = "lipid_metabolism"
    CARBOHYDRATE = "carbohydrate_metabolism"
    VITAMIN = "vitamin_metabolism"
    XENOBIOTIC = "xenobiotic_metabolism"


@dataclass
class TransgenerationalMetabolite:
    """
    A metabolite linked to transgenerational health effects.

    Attributes:
        name: Metabolite name
        pathway: Metabolic pathway
        transmission_line: Maternal, paternal, or both
        f1_effect: Effect observed in F1 generation
        f2_effect: Effect observed in F2 generation
        antagonism: Whether shows grandparental antagonism
    """

    name: str
    pathway: MetabolicPathway
    transmission_line: TransmissionLine
    f1_effect: Optional[str] = None
    f2_effect: Optional[str] = None
    antagonism: bool = False
    chds_study: str = "Hu ISEE 2025"


# Key transgenerational metabolites from CHDS
TRANSGENERATIONAL_METABOLITES = {
    # Amino acid metabolism - strong paternal effects
    "tryptophan": TransgenerationalMetabolite(
        name="Tryptophan",
        pathway=MetabolicPathway.AMINO_ACID,
        transmission_line=TransmissionLine.BOTH,
        f1_effect="age_at_menarche",
        f2_effect="age_at_menarche",
        antagonism=True,  # Grandpaternal vs grandmaternal show opposite effects
    ),
    "valine": TransgenerationalMetabolite(
        name="Valine",
        pathway=MetabolicPathway.AMINO_ACID,
        transmission_line=TransmissionLine.BOTH,
        f1_effect="obesity",
        f2_effect="obesity",
        antagonism=True,
    ),
    "leucine": TransgenerationalMetabolite(
        name="Leucine",
        pathway=MetabolicPathway.AMINO_ACID,
        transmission_line=TransmissionLine.BOTH,
        f2_effect="obesity",
        antagonism=True,
    ),
    "isoleucine": TransgenerationalMetabolite(
        name="Isoleucine",
        pathway=MetabolicPathway.AMINO_ACID,
        transmission_line=TransmissionLine.BOTH,
        f2_effect="obesity",
        antagonism=True,
    ),
    # Lipid metabolism
    "arachidonic_acid": TransgenerationalMetabolite(
        name="Arachidonic acid",
        pathway=MetabolicPathway.LIPID,
        transmission_line=TransmissionLine.BOTH,
        f2_effect="obesity",
    ),
    "omega_3": TransgenerationalMetabolite(
        name="Omega-3 fatty acid",
        pathway=MetabolicPathway.LIPID,
        transmission_line=TransmissionLine.MATERNAL,
        f2_effect="obesity_protective",
    ),
    # Carnitine shuttle - key for F2 obesity
    "carnitine": TransgenerationalMetabolite(
        name="Carnitine",
        pathway=MetabolicPathway.LIPID,
        transmission_line=TransmissionLine.BOTH,
        f2_effect="obesity",
    ),
}


# Paternal chemical exposures linked to offspring outcomes
PATERNAL_EXPOSURES = {
    "diisotridecyl_phthalate": {
        "name": "Diisotridecyl phthalate",
        "effect": "earlier_menarche",
        "generations_affected": ["F1", "F2"],
        "direction": "negative",  # Higher exposure → earlier menarche
        "note": "Chemical suspect from T3DB annotation",
    },
    "phenoxyethanol": {
        "name": "Phenoxyethanol",
        "effect": "earlier_menarche",
        "generations_affected": ["F1", "F2"],
        "direction": "negative",
        "note": "Chemical suspect from T3DB annotation",
    },
}


# Obesity progression across generations
OBESITY_PROGRESSION = {
    "F0_grandmother": {
        "lean_pct": 83,
        "overweight_pct": 11,
        "obese_pct": 6,
        "mean_bmi": {"lean": 21, "overweight": 27, "obese": 37},
    },
    "F1_daughter": {
        "lean_pct": 66,
        "overweight_pct": 19,
        "obese_pct": 15,
        "mean_bmi": {"lean": 22, "overweight": 27, "obese": 35},
        "years_later": "17-42",
    },
    "F2_granddaughter": {
        "lean_pct": 65,
        "overweight_pct": 16,
        "obese_pct": 19,
        "mean_bmi": {"lean": 21, "overweight": 27, "obese": 38},
        "years_later": "16-35",
    },
}


# Pathways with differential maternal vs paternal effects
DIFFERENTIAL_PATHWAYS = {
    # Stronger grandmaternal effect on F1
    "F1_maternal_dominant": [
        "Androgen and estrogen",
        "C21-steroid hormone",
        "Bile acid biosynthesis",
        "Arachidonic acid",
    ],
    # Stronger grandpaternal effect on F2 (key finding!)
    "F2_paternal_dominant": [
        "Urea cycle/amino group",
        "Arginine and Proline",
        "Valine, Leucine, Isoleucine",
        "Bile acid biosynthesis",
        "Omega-3 fatty acid",
        "Glycerophospholipid",
        "Fatty acid beta-oxidation",
        "Xenobiotics",
        "Drug metabolism (CYP450)",
    ],
}


class PaternalTransmissionModel:
    """
    Model for assessing paternal/grandparental transmission of health risks.

    Key finding: Paternal sperm can adjust offspring biology in response
    to environment, with effects potentially stronger in F2 than F1.

    Example:
        >>> model = PaternalTransmissionModel()
        >>> risk = model.assess_transgenerational_risk(
        ...     f0_paternal_tryptophan="high",
        ...     f0_maternal_tryptophan="low",
        ...     target_outcome="early_menarche"
        ... )
    """

    def __init__(self):
        """Initialize paternal transmission model."""
        self.metabolites = TRANSGENERATIONAL_METABOLITES
        self.exposures = PATERNAL_EXPOSURES

    def get_paternal_dominant_pathways(self) -> list[str]:
        """Get pathways where paternal effects dominate in F2."""
        return DIFFERENTIAL_PATHWAYS["F2_paternal_dominant"]

    def assess_antagonism(
        self,
        metabolite: str,
        f0_paternal_level: str,
        f0_maternal_level: str,
    ) -> dict:
        """
        Assess grandparental antagonism for a metabolite.

        When maternal metabolite is low, paternal contribution is stronger.

        Args:
            metabolite: Name of metabolite
            f0_paternal_level: "high", "medium", or "low"
            f0_maternal_level: "high", "medium", or "low"

        Returns:
            Antagonism assessment
        """
        met = self.metabolites.get(metabolite.lower())
        if not met or not met.antagonism:
            return {"has_antagonism": False}

        # Key finding: When maternal is low, paternal effect is stronger
        paternal_dominant = f0_maternal_level == "low"

        return {
            "metabolite": metabolite,
            "has_antagonism": True,
            "paternal_dominant": paternal_dominant,
            "mechanism": (
                "When essential metabolite in maternal serum is low, "
                "there is stronger contribution from grandfather to drive "
                "granddaughter's outcome (through epigenetic changes)."
            ),
            "affected_outcomes": [met.f1_effect, met.f2_effect],
        }

    def predict_f2_obesity_risk(
        self,
        f0_grandmother_bmi_category: str,
        f0_paternal_valine: str = "medium",
        f0_maternal_valine: str = "medium",
    ) -> dict:
        """
        Predict F2 granddaughter obesity risk.

        Key finding: Lean grandmothers (F0) had children and grandchildren
        who develop obesity.

        Args:
            f0_grandmother_bmi_category: "lean", "overweight", or "obese"
            f0_paternal_valine: Grandfather valine level
            f0_maternal_valine: Grandmother valine level

        Returns:
            Obesity risk prediction
        """
        base_progression = OBESITY_PROGRESSION

        # Different pathways for lean vs obese grandmothers
        if f0_grandmother_bmi_category == "lean":
            relevant_pathways = [
                "Androgen and estrogen",
                "Caffeine",
                "N-Glycan",
                "Carnitine shuttle",
                "Fructose and mannose",
            ]
        else:
            relevant_pathways = [
                "Tyrosine",
                "Bile acid biosynthesis",
                "Carnitine shuttle",
                "Leukotriene",
                "Arachidonic acid",
                "Prostaglandin formation",
                "Anti-inflammatory metabolites",
                "Omega-3 fatty acid",
            ]

        # Valine antagonism affects F2 obesity
        antagonism = self.assess_antagonism(
            "valine", f0_paternal_valine, f0_maternal_valine
        )

        return {
            "f0_bmi_category": f0_grandmother_bmi_category,
            "f2_obesity_increase": "83% lean F0 → 19% obese F2",
            "relevant_pathways": relevant_pathways,
            "valine_antagonism": antagonism,
            "key_finding": (
                "Lean women had children and grandchildren who develop obesity. "
                "Grandfather's metabolism interacts with grandmother's in "
                "predicting F2 obesity."
            ),
        }

    def get_key_conclusions(self) -> list[str]:
        """Return key conclusions from CHDS three-generation study."""
        return [
            "Male line (paternal/grandpaternal) plays unrecognized but critical role "
            "in shaping offspring reproductive health.",
            "Paternal sperm shows ability to adjust offspring biology in response "
            "to environment.",
            "Paternal and maternal metabolites interact, with potentially stronger "
            "generational effects seen in F2 than F1.",
            "When essential metabolite in maternal serum is low, there is "
            "consistently stronger contribution from grandfather to drive "
            "granddaughter's age at menarche and obesity.",
            "Paternal exposure to environmental chemicals (phthalates, phenoxyethanol) "
            "may affect offspring reproductive timing across generations.",
        ]

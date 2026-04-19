"""
Protective Factors Against Breast Cancer

CHDS findings on metabolites and placental factors that PROTECT against breast cancer.
Based on Cirillo et al. SOT 2026 poster - MFI and nicotinamide protection.

Key findings:
- Maternal Floor Infarction (MFI) protects in both F0 (mothers) and F1 (daughters)
- Nicotinamide, icosanoyl, urate, L-fucose-1-P are protective metabolites
- These pathways represent potential chemoprevention targets
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ProtectivePathway(Enum):
    """Protective pathway categories from CHDS research."""

    VITAMIN_B3 = "vitamin_b3_nicotinamide"
    FATTY_ACID = "phytanic_acid_peroxisomal"
    PURINE = "purine_metabolism"
    GLYCAN = "glycosphingolipid_biosynthesis"
    PLACENTAL = "placental_invasion_resistance"


@dataclass
class ProtectiveMetabolite:
    """
    A metabolite associated with breast cancer PROTECTION in CHDS.

    Attributes:
        name: Metabolite name
        pathway: Associated pathway
        hazard_ratio: HR < 1 indicates protection
        ci_95: 95% confidence interval
        mfi_associated: Whether associated with MFI pathology
        chds_study: Reference
    """

    name: str
    pathway: ProtectivePathway
    hazard_ratio: float
    ci_95: tuple[float, float]
    mfi_associated: bool
    chds_study: str = "Cirillo SOT 2026"


# CHDS protective metabolites from MWAS
CHDS_PROTECTIVE_METABOLITES = {
    "nicotinamide": ProtectiveMetabolite(
        name="Nicotinamide",
        pathway=ProtectivePathway.VITAMIN_B3,
        hazard_ratio=0.72,
        ci_95=(0.50, 1.04),
        mfi_associated=True,
    ),
    "icosanoyl": ProtectiveMetabolite(
        name="Icosanoyl",
        pathway=ProtectivePathway.FATTY_ACID,
        hazard_ratio=0.73,
        ci_95=(0.54, 0.98),
        mfi_associated=True,
    ),
    "urate": ProtectiveMetabolite(
        name="Urate",
        pathway=ProtectivePathway.PURINE,
        hazard_ratio=0.63,
        ci_95=(0.44, 0.90),
        mfi_associated=True,
    ),
    "l_fucose_1_p": ProtectiveMetabolite(
        name="L-fucose-1-phosphate",
        pathway=ProtectivePathway.GLYCAN,
        hazard_ratio=0.63,
        ci_95=(0.45, 0.88),
        mfi_associated=True,
    ),
}


# MFI protection across generations
MFI_PROTECTION = {
    "F0_mothers": {
        "hazard_ratio": 0.40,
        "ci_95": (0.18, 0.88),
        "study": "Cohn 2001 JNCI",
    },
    "F1_daughters": {
        "hazard_ratio": 0.39,
        "ci_95": (0.15, 0.99),
        "study": "Cirillo 2020 Reprod Toxicol",
    },
}


# Pathways enriched in MFI pregnancies
MFI_ENRICHED_PATHWAYS = [
    "Phytanic acid peroxisomal oxidation",
    "Glycine, serine, alanine and threonine metabolism",
    "Carnitine shuttle",
    "Lysine metabolism",
    "Vitamin D",
    "Glycosphingolipid biosynthesis - neolactoseries",
    "Lipoate metabolism",
    "Vitamin H (biotin) metabolism",
    "Blood Group Biosynthesis",
    "Valine, leucine and isoleucine degradation",
    "Purine metabolism",
    "Limonene and pinene degradation",
    "TCA cycle",
    "Chondroitin sulfate degradation",
    "Vitamin B3 (nicotinate and nicotinamide) metabolism",
    "Squalene and cholesterol biosynthesis",
    "Aspartate and asparagine metabolism",
    "De novo fatty acid biosynthesis",
]


class ProtectiveFactorModel:
    """
    Model for assessing protective factors against breast cancer.

    The placenta mimics hallmark traits of malignant tumor cells (proliferation,
    migration, invasion) but unlike cancer cells, assumes quiescence through
    senescence and apoptosis. MFI represents exaggerated invasion resistance.

    Example:
        >>> model = ProtectiveFactorModel()
        >>> protection = model.assess_protection(
        ...     has_mfi=True,
        ...     nicotinamide_tertile=3
        ... )
    """

    def __init__(self):
        """Initialize protective factor model."""
        self.metabolites = CHDS_PROTECTIVE_METABOLITES
        self.mfi_protection = MFI_PROTECTION

    def get_all_protective_metabolites(self) -> list[ProtectiveMetabolite]:
        """Get all protective metabolites."""
        return list(self.metabolites.values())

    def assess_protection(
        self,
        has_mfi: bool = False,
        metabolite_tertiles: Optional[dict[str, int]] = None,
    ) -> dict:
        """
        Assess breast cancer protection profile.

        Args:
            has_mfi: Whether pregnancy had MFI pathology
            metabolite_tertiles: Dict mapping metabolite names to tertile (1, 2, or 3)

        Returns:
            Protection assessment with contributing factors
        """
        protective_factors = []
        combined_hr = 1.0

        # MFI contribution
        if has_mfi:
            protective_factors.append({
                "factor": "Maternal Floor Infarction",
                "type": "placental",
                "hazard_ratio": 0.40,
                "mechanism": "Exaggerated invasion resistance via TGF-β/Smad pathway",
            })
            combined_hr *= 0.40

        # Metabolite contributions
        if metabolite_tertiles:
            for name, tertile in metabolite_tertiles.items():
                if name in self.metabolites and tertile == 3:  # High tertile
                    met = self.metabolites[name]
                    protective_factors.append({
                        "factor": met.name,
                        "type": "metabolite",
                        "hazard_ratio": met.hazard_ratio,
                        "pathway": met.pathway.value,
                    })
                    combined_hr *= met.hazard_ratio

        # Calculate protection level
        if combined_hr < 0.5:
            protection_level = "high"
        elif combined_hr < 0.75:
            protection_level = "moderate"
        else:
            protection_level = "baseline"

        return {
            "n_protective_factors": len(protective_factors),
            "combined_hazard_ratio": round(combined_hr, 3),
            "protection_level": protection_level,
            "protective_factors": protective_factors,
            "recommendations": self._get_recommendations(protective_factors),
        }

    def _get_recommendations(self, factors: list[dict]) -> list[str]:
        """Generate recommendations based on protective profile."""
        recommendations = []

        factor_types = [f["factor"] for f in factors]

        if "Nicotinamide" in factor_types:
            recommendations.append(
                "Nicotinamide has demonstrated anti-tumor activity. "
                "Consider as potential chemoprevention agent."
            )

        if "L-fucose-1-phosphate" in factor_types:
            recommendations.append(
                "Fluorinated fucose 1-phosphates (synthetic L-fucose-1-P analogs) "
                "reduce cancer cell proliferation and are in Phase I trials."
            )

        if any("MFI" in f for f in factor_types):
            recommendations.append(
                "MFI indicates enhanced invasion resistance pathways. "
                "TGF-β/Smad pathway may be protective mechanism."
            )

        if not recommendations:
            recommendations.append(
                "Standard breast cancer prevention strategies apply."
            )

        return recommendations

    def get_therapeutic_targets(self) -> list[dict]:
        """
        Get potential therapeutic targets from protective pathways.

        Returns list of druggable targets based on CHDS protective findings.
        """
        return [
            {
                "metabolite": "Nicotinamide",
                "target_pathway": "NAD+ biosynthesis",
                "existing_drugs": ["Nicotinamide (vitamin B3)", "NAD+ precursors"],
                "development_stage": "available",
                "rationale": "HR 0.72 for breast cancer in CHDS",
            },
            {
                "metabolite": "L-fucose-1-phosphate",
                "target_pathway": "Fucosylation",
                "existing_drugs": ["Fluorinated fucose analogs"],
                "development_stage": "phase1",
                "rationale": "Reduces cancer cell proliferation and migration",
            },
            {
                "metabolite": "Icosanoyl",
                "target_pathway": "Phytanic acid peroxisomal oxidation",
                "existing_drugs": [],
                "development_stage": "preclinical",
                "rationale": "Anti-tumor activity demonstrated",
            },
        ]

    def get_mfi_biology(self) -> dict:
        """Return information about MFI biology relevant to cancer."""
        return {
            "description": (
                "Maternal Floor Infarction (MFI) is characterized by marked fibrin "
                "deposition on the maternal surface arising from abnormal host-placenta "
                "interactions involving angiogenic and inflammatory processes."
            ),
            "cancer_parallel": (
                "The human placenta mimics hallmark traits of malignant tumor cells "
                "(proliferation, migration, invasion). Unlike cancer cells, placental "
                "cells ultimately assume quiescence through senescence and apoptosis."
            ),
            "protective_mechanism": (
                "MFI represents exaggerated invasion resistance, suggesting that "
                "anti-invasion pathways (TGF-β/Smad) are upregulated."
            ),
            "multigenerational": (
                "Protection extends to daughters (F1), suggesting epigenetic programming "
                "of anti-cancer pathways during fetal development."
            ),
        }

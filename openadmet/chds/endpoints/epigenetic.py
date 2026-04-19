"""
Epigenetic Memory: DNA Methylation Across Generations

CHDS findings on how ancestral exposures echo through generations via DNAm.
Based on Kuodza et al. SOT 2026 poster - UC Davis LaSalle Lab collaboration.

Key findings:
- DNAm patterns in F1/F2 are associated with F0 o,p'-DDT exposure
- 4 modules linked to F1 overweight + F2 obesity + F0 DDT
- 2 modules linked to F1/F2 menarche + F1 breast cancer + F0 exposures
- Enriched in cancer and hormone signaling pathways (KEGG)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DNAmModule(Enum):
    """DNA methylation network modules from WGCNA analysis."""

    YELLOW = "yellow"      # 391 regions - obesity, DDT
    PINK = "pink"          # 25 regions - obesity, DDT
    BROWN = "brown"        # 1631 regions - obesity, DDT
    TAN = "tan"            # 17 regions - F0 obesity, F1 overweight, F2 obesity
    SALMON = "salmon"      # 11 regions - menarche, breast cancer
    MIDNIGHTBLUE = "midnightblue"  # 10 regions - early menarche, breast cancer


@dataclass
class EpigeneticModule:
    """
    A DNA methylation network module from CHDS.

    Attributes:
        name: Module color name
        n_regions: Number of genomic regions
        top_genes: Top genes in the module
        f1_associations: Phenotypes associated in F1
        f2_associations: Phenotypes associated in F2
        f0_exposures: F0 exposures linked to this module
    """

    name: DNAmModule
    n_regions: int
    top_genes: list[str]
    f1_associations: list[str]
    f2_associations: list[str]
    f0_exposures: list[str]
    chds_study: str = "Kuodza SOT 2026"


# DNAm modules from CHDS three-generation study
DNAM_MODULES = {
    DNAmModule.YELLOW: EpigeneticModule(
        name=DNAmModule.YELLOW,
        n_regions=391,
        top_genes=[
            "AVEN",        # Apoptosis regulator
            "ARHGAP10",    # GTPase activating
            "PTPRO",       # Protein tyrosine phosphatase
            "AFF2",        # AF4/FMR2 family
            "TMEM202-AS1",
            "LRBA",        # LPS-responsive beige-like anchor
            "PCMTD1",
            "RASSF9",      # Ras association domain
            "PDLIM3",
            "AR",          # ANDROGEN RECEPTOR - key for hormone signaling!
        ],
        f1_associations=["overweight"],
        f2_associations=["obesity"],
        f0_exposures=["o,p'-DDT"],
    ),
    DNAmModule.PINK: EpigeneticModule(
        name=DNAmModule.PINK,
        n_regions=25,
        top_genes=["LOC124903068", "LOC124902916"],
        f1_associations=["overweight"],
        f2_associations=["obesity"],
        f0_exposures=["o,p'-DDT"],
    ),
    DNAmModule.BROWN: EpigeneticModule(
        name=DNAmModule.BROWN,
        n_regions=1631,
        top_genes=[
            "EXOC5",       # Exocyst complex
            "NARS2",       # Mitochondrial asparaginyl-tRNA synthetase
            "PAWR",        # Pro-apoptotic
            "IQCJ-SCHIP1",
            "USP32",       # Ubiquitin specific peptidase
            "ANKRD45",
            "LINC01762",
            "LOC124903974",
            "LOC105376449",
        ],
        f1_associations=["overweight"],
        f2_associations=["obesity"],
        f0_exposures=["o,p'-DDT"],
    ),
    DNAmModule.TAN: EpigeneticModule(
        name=DNAmModule.TAN,
        n_regions=17,
        top_genes=[
            "OR5AC2",      # Olfactory receptor
            "OR5H1",
            "OR5H14",
            "OR5H15",
            "LOC105373997",
            "LOC105373999",
        ],
        f1_associations=["overweight"],
        f2_associations=["obesity"],
        f0_exposures=["F0_obesity"],  # Links grandmother obesity to grandchild
    ),
    DNAmModule.SALMON: EpigeneticModule(
        name=DNAmModule.SALMON,
        n_regions=11,
        top_genes=["LAMC1"],  # Laminin - extracellular matrix, cancer-relevant
        f1_associations=["age_at_menarche", "breast_cancer"],
        f2_associations=["age_at_menarche"],
        f0_exposures=["PCB-28", "PCB-170", "Oxy", "PFHxA", "PFOA"],
    ),
    DNAmModule.MIDNIGHTBLUE: EpigeneticModule(
        name=DNAmModule.MIDNIGHTBLUE,
        n_regions=10,
        top_genes=[
            "POT1",        # Telomere protection! Cancer-relevant
            "POT1-AS1",
        ],
        f1_associations=["early_menarche", "breast_cancer"],
        f2_associations=["early_menarche", "age_at_menarche"],
        f0_exposures=["o,p'-DDT"],
    ),
}


# KEGG pathway enrichment from DNAm modules
KEGG_ENRICHMENT = {
    "obesity_modules": [
        "Breast cancer",
        "Prostate cancer",
        "Endocrine resistance",
        "Estrogen signaling pathway",
        "Thyroid hormone signaling pathway",
        "Oxytocin signaling pathway",
    ],
}


# GWAS catalog overlap - DNAm modules overlap with genetic risk loci
GWAS_OVERLAP = [
    "Body mass index",
    "Waist-hip ratio",
    "Type 2 diabetes",
    "Breast cancer",
    "Age at menarche",
    "Reproductive timing",
]


class EpigeneticMemoryModel:
    """
    Model for assessing epigenetic transmission of exposure effects.

    Demonstrates how F0 exposures create "epigenetic memory" that
    persists across F1 and F2 generations via DNA methylation.

    Example:
        >>> model = EpigeneticMemoryModel()
        >>> transmission = model.assess_epigenetic_transmission(
        ...     f0_exposure="o,p'-DDT",
        ...     outcome="obesity"
        ... )
    """

    def __init__(self):
        """Initialize epigenetic memory model."""
        self.modules = DNAM_MODULES

    def get_modules_for_exposure(self, exposure: str) -> list[EpigeneticModule]:
        """Get DNAm modules associated with an F0 exposure."""
        matching = []
        for module in self.modules.values():
            if exposure in module.f0_exposures:
                matching.append(module)
        return matching

    def get_modules_for_outcome(
        self, outcome: str, generation: str = "F2"
    ) -> list[EpigeneticModule]:
        """Get DNAm modules associated with an outcome."""
        matching = []
        for module in self.modules.values():
            if generation == "F1" and outcome in module.f1_associations:
                matching.append(module)
            elif generation == "F2" and outcome in module.f2_associations:
                matching.append(module)
        return matching

    def assess_epigenetic_transmission(
        self,
        f0_exposure: str,
        outcome: str,
    ) -> dict:
        """
        Assess epigenetic transmission from F0 exposure to F1/F2 outcome.

        Args:
            f0_exposure: The grandmaternal exposure (e.g., "o,p'-DDT")
            outcome: The health outcome (e.g., "obesity", "breast_cancer")

        Returns:
            Assessment of epigenetic transmission pathway
        """
        exposure_modules = self.get_modules_for_exposure(f0_exposure)
        outcome_modules_f1 = self.get_modules_for_outcome(outcome, "F1")
        outcome_modules_f2 = self.get_modules_for_outcome(outcome, "F2")

        # Find modules that link exposure AND outcome
        overlapping = []
        for mod in exposure_modules:
            if mod in outcome_modules_f1 or mod in outcome_modules_f2:
                overlapping.append(mod)

        if not overlapping:
            return {
                "f0_exposure": f0_exposure,
                "outcome": outcome,
                "epigenetic_link": False,
                "message": "No DNAm modules found linking this exposure-outcome pair.",
            }

        # Extract key genes
        key_genes = []
        for mod in overlapping:
            key_genes.extend(mod.top_genes[:3])

        return {
            "f0_exposure": f0_exposure,
            "outcome": outcome,
            "epigenetic_link": True,
            "n_dnam_modules": len(overlapping),
            "modules": [mod.name.value for mod in overlapping],
            "total_regions": sum(mod.n_regions for mod in overlapping),
            "key_genes": list(set(key_genes)),
            "f1_affected": any(
                outcome in mod.f1_associations for mod in overlapping
            ),
            "f2_affected": any(
                outcome in mod.f2_associations for mod in overlapping
            ),
            "mechanism": self._get_mechanism(overlapping),
        }

    def _get_mechanism(self, modules: list[EpigeneticModule]) -> str:
        """Infer mechanism from module genes."""
        all_genes = []
        for mod in modules:
            all_genes.extend(mod.top_genes)

        if "AR" in all_genes:
            return (
                "Androgen receptor (AR) methylation changes may mediate "
                "hormone-dependent effects across generations."
            )
        elif "POT1" in all_genes:
            return (
                "POT1 (telomere protection) methylation may affect cellular "
                "aging and cancer susceptibility across generations."
            )
        elif "LAMC1" in all_genes:
            return (
                "LAMC1 (laminin) methylation affects extracellular matrix, "
                "potentially influencing tissue development and cancer risk."
            )
        else:
            return (
                "DNA methylation changes at multiple loci create epigenetic "
                "memory that persists across generations."
            )

    def get_ddt_obesity_pathway(self) -> dict:
        """
        Get the specific DDT → obesity epigenetic pathway.

        This is a key CHDS finding: o,p'-DDT exposure in F0 pregnancy
        is linked to F1 overweight and F2 obesity through shared DNAm.
        """
        ddt_modules = self.get_modules_for_exposure("o,p'-DDT")
        obesity_modules = [
            mod for mod in ddt_modules
            if "obesity" in mod.f2_associations or "overweight" in mod.f1_associations
        ]

        return {
            "exposure": "o,p'-DDT (grandmaternal, 1959-1967)",
            "f1_outcome": "Overweight in daughters (~age 50)",
            "f2_outcome": "Obesity in granddaughters",
            "n_shared_modules": len(obesity_modules),
            "modules": [mod.name.value for mod in obesity_modules],
            "total_dnam_regions": sum(mod.n_regions for mod in obesity_modules),
            "key_genes": ["AR", "AVEN", "PTPRO", "PAWR"],
            "kegg_pathways": KEGG_ENRICHMENT["obesity_modules"],
            "gwas_overlap": ["Body mass index", "Waist-hip ratio"],
            "hypothesis": (
                "H1: DNAm patterns associated with obesity and breast cancer "
                "in F1/F2 are linked to grandmaternal o,p'-DDT exposure."
            ),
            "conclusion": (
                "Specific DNAm patterns altered in F1 and F2 are associated "
                "with F1 overweight, F2 obesity, and F0 o,p'-DDT exposure, "
                "demonstrating epigenetic memory across generations."
            ),
        }

    def get_menarche_cancer_pathway(self) -> dict:
        """
        Get the menarche → breast cancer epigenetic pathway.

        Early menarche is a breast cancer risk factor; this pathway
        shows F0 exposures → F1/F2 menarche → F1 breast cancer.
        """
        menarche_modules = [
            mod for mod in self.modules.values()
            if "age_at_menarche" in mod.f1_associations
            or "early_menarche" in mod.f1_associations
        ]

        return {
            "f0_exposures": ["o,p'-DDT", "PCB-28", "PCB-170", "PFOA", "PFHxA"],
            "f1_outcomes": ["age_at_menarche", "early_menarche", "breast_cancer"],
            "f2_outcomes": ["age_at_menarche", "early_menarche"],
            "n_modules": len(menarche_modules),
            "key_genes": ["LAMC1", "POT1", "POT1-AS1"],
            "pot1_relevance": (
                "POT1 protects telomeres; methylation changes may affect "
                "cellular aging, reproductive timing, and cancer susceptibility."
            ),
            "hypothesis": (
                "H2: DNAm patterns associated with early menarche and breast "
                "cancer in F1/F2 are linked to grandmaternal exposures."
            ),
        }

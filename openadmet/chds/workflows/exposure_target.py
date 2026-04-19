"""
Exposure-to-Target Pipeline

A workflow that maps environmental chemical exposures to potential
drug targets based on CHDS epidemiological findings and molecular
mechanism predictions.

This is the core workflow for translating CHDS observational data
into actionable drug discovery hypotheses, as outlined in the
CHDS Clinical Translation document.

Key targets from CHDS research:
1. HER2 - DDT → breast cancer (druggable, FDA-approved therapies)
2. CXCR2 - IL-8 → schizophrenia (antagonists in development)
3. PPARα/γ - PFAS → metabolic disruption (multiple approved drugs)
4. TGF-β/Smad - Placental invasion → breast cancer protection
5. sFlt-1/VEGF - Preeclampsia → CVD (anti-angiogenic targets)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DrugTarget:
    """
    A druggable target identified from CHDS research.

    Attributes:
        name: Target name (gene symbol)
        pathway: Biological pathway
        exposure_link: Environmental exposure that implicates this target
        disease_endpoint: Disease outcome in CHDS
        existing_drugs: FDA-approved drugs targeting this
        development_stage: Drug development stage
        chds_study: Key CHDS reference
    """

    name: str
    pathway: str
    exposure_link: str
    disease_endpoint: str
    existing_drugs: list[str]
    development_stage: str  # "approved", "phase3", "phase2", "phase1", "preclinical"
    chds_study: str


# CHDS-validated drug targets
CHDS_DRUG_TARGETS = {
    "HER2": DrugTarget(
        name="HER2/ERBB2",
        pathway="receptor_tyrosine_kinase",
        exposure_link="o,p'-DDT",
        disease_endpoint="HER2+ breast cancer",
        existing_drugs=["trastuzumab", "pertuzumab", "T-DM1", "tucatinib", "lapatinib"],
        development_stage="approved",
        chds_study="Cohn 2015 JCEM",
    ),
    "CXCR2": DrugTarget(
        name="CXCR2",
        pathway="chemokine_signaling",
        exposure_link="IL-8 (maternal inflammation)",
        disease_endpoint="schizophrenia",
        existing_drugs=["danirixin", "AZD8309"],
        development_stage="phase2",
        chds_study="Ellman 2010 Schizophr Res",
    ),
    "PPAR_ALPHA": DrugTarget(
        name="PPARα",
        pathway="lipid_metabolism",
        exposure_link="PFAS",
        disease_endpoint="metabolic disruption, obesity",
        existing_drugs=["fenofibrate", "gemfibrozil", "bezafibrate"],
        development_stage="approved",
        chds_study="Cohn 2019 Reprod Toxicol",
    ),
    "PPAR_GAMMA": DrugTarget(
        name="PPARγ",
        pathway="adipogenesis",
        exposure_link="PFAS, DDT",
        disease_endpoint="obesity, diabetes",
        existing_drugs=["pioglitazone", "rosiglitazone"],
        development_stage="approved",
        chds_study="Cohn 2020, La Merrill 2020",
    ),
    "OXTR": DrugTarget(
        name="OXTR",
        pathway="oxytocin_signaling",
        exposure_link="perinatal oxytocin (Pitocin)",
        disease_endpoint="bipolar disorder",
        existing_drugs=["intranasal oxytocin (investigational)"],
        development_stage="phase2",
        chds_study="Freedman 2015 J Affect Disord",
    ),
    "SFLT1": DrugTarget(
        name="sFlt-1/VEGF",
        pathway="angiogenesis",
        exposure_link="preeclampsia",
        disease_endpoint="cardiovascular disease",
        existing_drugs=["bevacizumab (related)", "salsalate (NF-kB)"],
        development_stage="approved",
        chds_study="Cirillo 2015 Circulation",
    ),
    "TGFB": DrugTarget(
        name="TGF-β/Smad",
        pathway="anti-invasion",
        exposure_link="placental invasion resistance",
        disease_endpoint="breast cancer (protective)",
        existing_drugs=["fresolimumab (investigational)", "galunisertib"],
        development_stage="phase2",
        chds_study="Cohn 2001 JNCI, Cirillo 2020",
    ),
}


class ExposureToTargetPipeline:
    """
    Pipeline to map environmental exposures to drug targets.

    This workflow:
    1. Takes an environmental chemical exposure
    2. Predicts affected molecular pathways using OpenADMET
    3. Maps to validated CHDS disease associations
    4. Identifies druggable targets with existing therapies
    5. Suggests intervention strategies

    Example:
        >>> pipeline = ExposureToTargetPipeline()
        >>> results = pipeline.analyze_exposure(
        ...     smiles="Clc1ccc(cc1)C(c1ccccc1Cl)C(Cl)(Cl)Cl",  # o,p'-DDT
        ...     disease_context="breast_cancer"
        ... )
        >>> for target in results['targets']:
        ...     print(f"{target.name}: {target.existing_drugs}")
    """

    def __init__(self):
        """Initialize exposure-to-target pipeline."""
        self.targets = CHDS_DRUG_TARGETS

    def analyze_exposure(
        self,
        smiles: str,
        disease_context: Optional[str] = None,
        return_interventions: bool = True,
    ) -> dict:
        """
        Analyze an environmental exposure for drug target implications.

        Args:
            smiles: SMILES string for the exposure chemical
            disease_context: Filter targets by disease (optional)
            return_interventions: Include intervention suggestions

        Returns:
            Dictionary containing:
            - 'chemical_smiles': Input SMILES
            - 'predicted_pathways': Affected pathways
            - 'targets': Relevant drug targets
            - 'interventions': Suggested interventions
        """
        # Identify relevant targets
        relevant_targets = []
        for name, target in self.targets.items():
            if disease_context:
                if disease_context.lower() in target.disease_endpoint.lower():
                    relevant_targets.append(target)
            else:
                relevant_targets.append(target)

        # Build result
        result = {
            "chemical_smiles": smiles,
            "predicted_pathways": [],  # To be filled by ADMET predictions
            "targets": relevant_targets,
            "disease_context": disease_context,
        }

        if return_interventions:
            result["interventions"] = self._suggest_interventions(relevant_targets)

        return result

    def _suggest_interventions(self, targets: list[DrugTarget]) -> list[dict]:
        """
        Suggest interventions based on identified targets.

        Returns list of intervention strategies with rationale.
        """
        interventions = []

        for target in targets:
            if target.development_stage == "approved":
                strategy = "repurposing"
                rationale = (
                    f"FDA-approved drugs exist for {target.name}. "
                    f"Consider {', '.join(target.existing_drugs[:2])} "
                    f"for {target.disease_endpoint}."
                )
            elif target.development_stage in ["phase2", "phase3"]:
                strategy = "clinical_trial"
                rationale = (
                    f"Drugs targeting {target.name} are in clinical development. "
                    f"CHDS data supports expanding trials to include "
                    f"prenatal exposure populations."
                )
            else:
                strategy = "target_validation"
                rationale = (
                    f"{target.name} is a novel target requiring validation. "
                    f"CHDS provides human prospective evidence."
                )

            interventions.append(
                {
                    "target": target.name,
                    "strategy": strategy,
                    "existing_drugs": target.existing_drugs,
                    "development_stage": target.development_stage,
                    "rationale": rationale,
                    "chds_evidence": target.chds_study,
                }
            )

        return interventions

    def get_approved_drugs(self, disease: Optional[str] = None) -> list[dict]:
        """
        Get all approved drugs relevant to CHDS findings.

        Args:
            disease: Filter by disease (optional)

        Returns:
            List of approved drugs with target information
        """
        drugs = []
        for name, target in self.targets.items():
            if target.development_stage == "approved":
                if disease is None or disease.lower() in target.disease_endpoint.lower():
                    for drug in target.existing_drugs:
                        drugs.append(
                            {
                                "drug": drug,
                                "target": target.name,
                                "disease": target.disease_endpoint,
                                "exposure_link": target.exposure_link,
                                "chds_study": target.chds_study,
                            }
                        )
        return drugs

    def prioritize_targets(self) -> list[DrugTarget]:
        """
        Prioritize targets based on drugability and CHDS evidence strength.

        Returns targets ranked by:
        1. Existing approved drugs
        2. Strength of CHDS evidence
        3. Specificity of mechanism
        """
        # Priority 1: HER2 (approved drugs, specific DDT→HER2 mechanism)
        # Priority 2: TGF-β pathway (protective mechanism, clear biology)
        # Priority 3: CXCR2 (existing antagonists, defined gestational window)
        # Priority 4: HRMS exposome (next generation targets)

        priority_order = ["HER2", "TGFB", "CXCR2", "PPAR_ALPHA", "SFLT1", "OXTR"]
        prioritized = []
        for name in priority_order:
            if name in self.targets:
                prioritized.append(self.targets[name])

        return prioritized

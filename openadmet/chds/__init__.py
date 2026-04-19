"""
CHDS (Child Health and Development Studies) Module for OpenADMET

This module extends OpenADMET with capabilities specific to:
- Developmental toxicity prediction
- Transplacental pharmacokinetics
- Multigenerational exposure modeling
- Pregnancy-specific ADMET endpoints
- Prostate cancer risk with race-specific pathways
- Protective factors against breast cancer
- Paternal/grandparental transmission mechanisms
- Epigenetic memory via DNA methylation

Based on 60+ years of prospective data from the CHDS cohort.

Key research areas supported:
- Environmental exposures (DDT, PCBs, PFAS) → disease outcomes
- Gestational biomarkers → offspring health
- Transgenerational epigenetic effects
- Developmental windows of susceptibility
- Three-generation metabolomics (F0 → F1 → F2)
- Race-specific metabolic pathways in cancer

References:
- Cohn BA et al. DDT and breast cancer (JNCI 2019)
- Go YM et al. Exposome epidemiology (Environ Int 2023)
- Cirillo PM et al. Pregnancy complications → CVD (Circulation 2015)
- Krigbaum et al. Prostate cancer MWAS (SOT 2026)
- Kuodza et al. Epigenetic memory across generations (SOT 2026)
- Hu et al. Three-generation metabolomics (ISEE 2025)
"""

from openadmet.chds.endpoints import (
    DevelopmentalToxicity,
    TransplacentalTransport,
    EndocrineDisruption,
    MultigenerationalRisk,
    ProstateCancerRisk,
    ProtectiveFactorModel,
    PaternalTransmissionModel,
    EpigeneticMemoryModel,
)
from openadmet.chds.data import CHDSExposomeLoader
from openadmet.chds.workflows import (
    ExposureToTargetPipeline,
    GestationalBiomarkerModel,
)

__all__ = [
    "DevelopmentalToxicity",
    "TransplacentalTransport",
    "EndocrineDisruption",
    "MultigenerationalRisk",
    "ProstateCancerRisk",
    "ProtectiveFactorModel",
    "PaternalTransmissionModel",
    "EpigeneticMemoryModel",
    "CHDSExposomeLoader",
    "ExposureToTargetPipeline",
    "GestationalBiomarkerModel",
]

__version__ = "0.1.0"

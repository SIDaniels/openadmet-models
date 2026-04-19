"""
CHDS-specific ADMET endpoints for developmental and reproductive toxicology.

These endpoints extend standard ADMET predictions to include:
- Developmental toxicity across gestational windows
- Transplacental transport and fetal exposure
- Endocrine disruption pathways (ER, AR, PPARs, etc.)
- Multigenerational risk modeling
- Prostate cancer with race-specific metabolic pathways
- Protective factors (nicotinamide, MFI)
- Paternal/grandparental transmission mechanisms
- Epigenetic memory via DNA methylation
"""

from openadmet.chds.endpoints.developmental import DevelopmentalToxicity
from openadmet.chds.endpoints.transplacental import TransplacentalTransport
from openadmet.chds.endpoints.endocrine import EndocrineDisruption
from openadmet.chds.endpoints.multigenerational import MultigenerationalRisk
from openadmet.chds.endpoints.prostate_cancer import ProstateCancerRisk
from openadmet.chds.endpoints.protective_factors import ProtectiveFactorModel
from openadmet.chds.endpoints.paternal_transmission import PaternalTransmissionModel
from openadmet.chds.endpoints.epigenetic import EpigeneticMemoryModel

__all__ = [
    "DevelopmentalToxicity",
    "TransplacentalTransport",
    "EndocrineDisruption",
    "MultigenerationalRisk",
    "ProstateCancerRisk",
    "ProtectiveFactorModel",
    "PaternalTransmissionModel",
    "EpigeneticMemoryModel",
]

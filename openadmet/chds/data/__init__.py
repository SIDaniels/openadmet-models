"""
CHDS Data Loading and Processing Module

Provides utilities for loading and processing CHDS exposome data,
including archived pregnancy serum measurements and clinical outcomes.

Data types supported:
- HRMS exposome profiles (Go et al. 2023)
- Targeted chemical measurements (DDT, PCBs, PFAS)
- Metabolomics signatures
- Clinical outcomes and disease endpoints
"""

from openadmet.chds.data.exposome import CHDSExposomeLoader

__all__ = [
    "CHDSExposomeLoader",
]

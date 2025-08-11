# UniMol Model Usage Guide

## Overview

The UniMol model is integrated into the OpenADMET framework for molecular property prediction tasks.

## Usage

### Classification Task

```bash
openadmet anvil --recipe-path openadmet/models/tests/unit/test_data/basic_anvil_unimol.yaml
```

### Regression Task

```bash
openadmet anvil --recipe-path openadmet/models/tests/unit/test_data/basic_anvil_unimol_regression.yaml
```

## YAML Configuration File Structure

### Basic Configuration

```yaml
metadata:
  version: "v1"
  driver: "pytorch"
  name: "unimol-model-example"
  description: "UniMol model for ADMET prediction"

data:
  type: "intake"
  resource: your_data.csv
  input_col: SMILES
  target_cols:
    - target_column_name

procedure:
  model:
    type: UniMolModel
    params:
      task: "classification"
      model_name: "unimolv1"
      model_size: "84m"
      epochs: 10
      batch_size: 16
      metrics: "auc"
      remove_hs: false


```

## Model Parameters

### UniMolModel Parameters

- `task`: Task type, "classification" or "regression"
- `model_name`: Model version, "unimolv1" or "unimolv2"
- `model_size`: Model size (only valid for unimolv2)
  - "84m": 84 million parameters
  - "164m": 164 million parameters
  - "310m": 310 million parameters
  - "570m": 570 million parameters
  - "1.1B": 1.1 billion parameters
- `epochs`: Number of training epochs
- `batch_size`: Batch size
- `metrics`: Evaluation metric
- `remove_hs`: Whether to remove hydrogen atoms

### Supported Evaluation Metrics

- Classification tasks: "auc"
- Regression tasks: "mse", "mae"

## Data Format Requirements

Input data should be a CSV file containing:
- SMILES column: SMILES representation of molecules
- Target column: Target values for prediction

Example:
```csv
SMILES,TARGET
CCO,1
CCC,0
CCCC,1
```

## Important Notes

1. Ensure the `unimol_tools` package is installed in the environment
2. UniMol models require significant computational resources, GPU is recommended
3. Large models (like 1.1B) require substantial memory
4. Training may take considerable time, recommend starting with smaller models for testing

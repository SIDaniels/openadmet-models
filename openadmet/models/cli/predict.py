
import click
import pandas as pd
from rdkit.Chem import PandasTools
from openadmet.models.anvil.workflow import ProcedureSpec
from pathlib import Path
from loguru import logger



def load_anvil_model_and_metadata(model_dir):
    """Load the Anvil model from the specified path"""
    # load from recipie directory
    recipie_components_dir = model_dir / "recipe_components"
    if not recipie_components_dir.exists():
        raise ValueError(f"Model path {model_dir} does not contain recipe components")
    # load the specification
    procedure_spec = recipie_components_dir / "procedure.yaml"
    if not procedure_spec.exists():
        raise ValueError(f"Model path {model_dir} does not contain procedure.yaml")


    # load the procedure specification
    procedure_spec = ProcedureSpec.from_yaml(procedure_spec)
    feat = procedure_spec.feat.to_class()
    model = procedure_spec.model.to_class()

    # deserialize the model
    loaded_model = model.deserialize(
        param_path=model_dir / model._model_json_name,
        serial_path=model_dir / model._model_save_name,
    )

    return loaded_model, feat



@click.command()

@click.option(
    "--input-path",
    help="Path to the input CSV file or SDF containing structureas",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--input-col",
    help="Column name in the CSV file containing input structure or SMILES",
    default="SMILES",
    show_default=True,
)
@click.option(
    "--model-dir",
    help="Path to the trained model directory as trained by `openadmet anvil`",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--output-path",
    help="Path to the output CSV file for predictions",
    default="predictions.csv",
    show_default=True,
    required=True,
    type=click.Path(exists=False, writable=True),
)
@click.option(
    "--debug", is_flag=True, help="Enable debug mode"
)
def predict(input_path, input_col,  model_dir, output_path, debug):
    """Predict using a trained model"""

    logger.info("Starting prediction")
    logger.debug(f"Input path: {input_path}")
    logger.debug(f"Model directory: {model_dir}")
    logger.debug(f"Output path: {output_path}")
    logger.debug(f"SMILES column: {input_col}")
    # load input data
    if input_path.endswith(".csv"):
        data = pd.read_csv(input_path)
    elif input_path.endswith(".sdf"):
        data = PandasTools.LoadSDF(input_path, smilesName=input_col)

    if input_col not in data.columns:
        raise ValueError(f"Column {input_col} not found in input data")
    

    if "predictions" in data.columns:
        raise ValueError("Output file already contains a 'predictions' column")

    # Load the model
    model, feat = load_anvil_model_and_metadata(Path(model_dir))

    logger.info(f"Model: {model}")
    logger.info(f"Feature: {feat}")
    X_feat, _ = feat.featurize(data[input_col])

    predictions = model.predict(X_feat)

    # todo metadata

    # Save predictions to output file
    data["predictions"] = predictions
    data.to_csv(output_path, index=False)



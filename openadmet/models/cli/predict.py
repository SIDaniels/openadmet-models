
import click
import pandas as pd
from rdkit.Chem import PandasTools
from openadmet.models.anvil.workflow import ProcedureSpecification

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
    procedure_spec = ProcedureSpecification.from_yaml(procedure_spec)
    model = procedure_spec.model.to_class()

    # deserialize the model
    loaded_model = model.deserialize(
        param_path=model_dir / model._model_json_name,
        serial_path=recipie_components_dir / model._model_save_name,
    )

    return loaded_model



@click.command()

@click.option(
    "--input-path",
    help="Path to the input CSV file or SDF containing structureas",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--smiles-col",
    help="Column name in the CSV file containing SMILES strings",
    required=True,
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
    required=True,
    type=click.Path(),
)
@click.option(
    "--debug", is_flag=True, help="Enable debug mode"
)
def predict(input_path, smiles_col,  model_dir, output_path, debug):
    """Predict using a trained model"""
    # load input data
    if input_path.endswith(".csv"):
        data = pd.read_csv(input_path)
    elif input_path.endswith(".sdf"):
        data = PandasTools.LoadSDF(input_path, smilesName=smiles_col)

    if smiles_col not in data.columns:
        raise ValueError(f"Column {smiles_col} not found in input data")
    
    # Load the model
    model = load_anvil_model_and_metadata(model_dir)

    # todo metadata


    predictions = model.predict(data[smiles_col])

    # Save predictions to output file
    data["predictions"] = predictions
    if output_path.endswith(".csv"):
        data.to_csv(output_path, index=False)
    elif output_path.endswith(".sdf"):
        PandasTools.SaveSDF(data, output_path, smilesCol=smiles_col, properties=["predictions"])
    else:
        raise ValueError("Output file must be either CSV or SDF format")


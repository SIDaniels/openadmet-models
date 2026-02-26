import numpy as np
import pandas as pd
import pytest

from openadmet.models.anvil.specification import (
    AnvilSpecification,
)
from openadmet.models.tests.unit.datafiles import (
    acetylcholinesterase_anvil_chemprop_yaml,
    anvil_yaml_featconcat,
    anvil_yaml_gridsearch,
    anvil_yaml_xgboost_cv,
    basic_anvil_yaml,
    basic_anvil_yaml_classification,
    basic_anvil_yaml_cv,
    tabpfn_anvil_classification_yaml,
)


def all_anvil_full_recipes():
    """Return a list of full anvil recipes for testing."""
    return [
        basic_anvil_yaml,
        # anvil_yaml_featconcat, # skipping as slow, redundant with integration tests
        anvil_yaml_gridsearch,
        # anvil_yaml_xgboost_cv, # skipping as slow, redundant with integration tests
    ]


def test_anvil_spec_create():
    """Test creating an AnvilSpecification from a YAML recipe file."""
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    assert anvil_spec


def test_anvil_spec_create_from_recipe_roundtrip(tmp_path):
    """
    Test the round-trip serialization of AnvilSpecification (load -> save -> load).

    This ensures that the specification object can be correctly serialized to YAML and deserialized back,
    preserving all configuration settings.
    """
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    assert anvil_spec
    anvil_spec.to_recipe(tmp_path / "tst.yaml")
    anvil_spec2 = AnvilSpecification.from_recipe(tmp_path / "tst.yaml")
    # these were created from different directories, so the anvil_dir will be different
    anvil_spec.data.anvil_dir = None
    anvil_spec2.data.anvil_dir = None

    assert anvil_spec == anvil_spec2


def test_anvil_spec_create_to_workflow():
    """Test converting a specification into an executable Workflow object."""
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    anvil_workflow = anvil_spec.to_workflow()
    assert anvil_workflow


@pytest.mark.parametrize("anvil_full_recipie", all_anvil_full_recipes())
def test_anvil_workflow_run(tmp_path, anvil_full_recipie, mocker):
    """
    Test running a full Anvil workflow with mocked training and data components.

    This test verifies that the workflow orchestration logic correctly calls:
    - Data loading
    - Splitting
    - Featurization
    - Model training
    - Serialization

    We mock heavy components (train, read, featurize) to make this a fast unit test rather than a slow integration test.
    """
    anvil_workflow = AnvilSpecification.from_recipe(anvil_full_recipie).to_workflow()
    X = pd.DataFrame({"smiles": ["CCO", "CCN"]})
    y = pd.DataFrame({"target": [1.0, 2.0]})
    train_spy = mocker.patch.object(type(anvil_workflow), "_train", autospec=True)
    mocker.patch.object(type(anvil_workflow.data_spec), "read", return_value=(X, y))
    mocker.patch.object(
        type(anvil_workflow.split),
        "split",
        return_value=(X, None, None, y, None, None, None),
    )
    mocker.patch.object(
        type(anvil_workflow.feat),
        "featurize",
        side_effect=[
            (np.array([[0.1], [0.2]]), None),
            (np.array([[0.1], [0.2]]), None),
        ],
    )
    mocker.patch.object(type(anvil_workflow.model), "serialize")
    mocker.patch("openadmet.models.anvil.workflow.zarr.save")
    anvil_workflow.run(output_dir=tmp_path / "tst")
    train_spy.assert_called_once()


def test_anvil_multiyaml(tmp_path):
    """
    Test splitting and recombining Anvil specifications into multiple YAML files.

    The Anvil system supports splitting config into metadata, procedure, data, and report files.
    This test ensures that splitting a spec and reloading it from parts yields the same object.
    """
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    anvil_spec.to_multi_yaml(
        metadata_yaml=tmp_path / "metadata.yaml",
        procedure_yaml=tmp_path / "procedure.yaml",
        data_yaml=tmp_path / "data.yaml",
        report_yaml=tmp_path / "eval.yaml",
    )
    anvil_spec2 = AnvilSpecification.from_multi_yaml(
        metadata_yaml=tmp_path / "metadata.yaml",
        procedure_yaml=tmp_path / "procedure.yaml",
        data_yaml=tmp_path / "data.yaml",
        report_yaml=tmp_path / "eval.yaml",
    )
    assert anvil_spec.data.anvil_dir == anvil_spec2.data.anvil_dir
    assert anvil_spec.dict() == anvil_spec2.dict()


def test_anvil_cross_val_run(tmp_path, mocker):
    """
    Test running a cross-validation Anvil workflow with mocked components.

    Ensures that the workflow correctly handles the cross-validation logic (though exact CV splitting
    is mocked here, the workflow structure is verified).
    """
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml_cv)
    anvil_workflow = anvil_spec.to_workflow()
    X = pd.DataFrame({"smiles": ["CCO", "CCN"]})
    y = pd.DataFrame({"target": [1.0, 2.0]})
    train_spy = mocker.patch.object(type(anvil_workflow), "_train", autospec=True)
    mocker.patch.object(type(anvil_workflow.data_spec), "read", return_value=(X, y))
    mocker.patch.object(
        type(anvil_workflow.split),
        "split",
        return_value=(X, None, None, y, None, None, None),
    )
    mocker.patch.object(
        type(anvil_workflow.feat),
        "featurize",
        side_effect=[
            (np.array([[0.1], [0.2]]), None),
            (np.array([[0.1], [0.2]]), None),
        ],
    )
    mocker.patch.object(type(anvil_workflow.model), "serialize")

    # TODO: verify because this looks wrong
    mocker.patch("openadmet.models.anvil.workflow.zarr.save")
    anvil_workflow.run(output_dir=tmp_path / "tst")
    train_spy.assert_called_once()


def test_anvil_classification_run(tmp_path, mocker):
    """
    Test running a classification Anvil workflow with mocked components.

    Verifies workflow execution for classification tasks (integer targets).
    """
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml_classification)
    anvil_workflow = anvil_spec.to_workflow()
    X = pd.DataFrame({"smiles": ["CCO", "CCN"]})
    y = pd.DataFrame({"target": [0, 1]})
    train_spy = mocker.patch.object(type(anvil_workflow), "_train", autospec=True)
    mocker.patch.object(type(anvil_workflow.data_spec), "read", return_value=(X, y))
    mocker.patch.object(
        type(anvil_workflow.split),
        "split",
        return_value=(X, None, None, y, None, None, None),
    )
    mocker.patch.object(
        type(anvil_workflow.feat),
        "featurize",
        side_effect=[
            (np.array([[0.1], [0.2]]), None),
            (np.array([[0.1], [0.2]]), None),
        ],
    )
    mocker.patch.object(type(anvil_workflow.model), "serialize")

    # TODO: verify because this looks wrong
    mocker.patch("openadmet.models.anvil.workflow.zarr.save")
    anvil_workflow.run(output_dir=tmp_path / "tst")
    train_spy.assert_called_once()


# skip on MacOS runner?
def test_anvil_chemprop_cpu_regression(tmp_path, mocker):
    """
    Test running a ChemProp (deep learning) workflow on CPU.

    Verifies that the workflow can handle ChemProp-specific logic (return values from featurizer, etc.).
    """
    anvil_spec = AnvilSpecification.from_recipe(
        acetylcholinesterase_anvil_chemprop_yaml
    )
    anvil_workflow = anvil_spec.to_workflow()
    X = pd.DataFrame({"smiles": ["CCO", "CCN"]})
    y = pd.DataFrame({"target": [1.0, 2.0]})
    train_spy = mocker.patch.object(type(anvil_workflow), "_train", autospec=True)
    mocker.patch.object(type(anvil_workflow.data_spec), "read", return_value=(X, y))
    mocker.patch.object(
        type(anvil_workflow.split),
        "split",
        return_value=(X, None, None, y, None, None, None),
    )
    mocker.patch.object(
        type(anvil_workflow.feat),
        "featurize",
        return_value=(object(), None, None, [0]),
    )
    mocker.patch.object(type(anvil_workflow.model), "serialize")

    # TODO: verify because this looks wrong
    mocker.patch("openadmet.models.anvil.workflow.torch.save")
    anvil_workflow.run(output_dir=tmp_path / "tst")
    train_spy.assert_called_once()


@pytest.mark.skip(reason="TabPFN requires GPU and is not supported on MacOS runners")
def test_anvil_tabpfn_classification(tmp_path):
    """Test TabPFN classification workflow (skipped on non-GPU environments)."""
    anvil_spec = AnvilSpecification.from_recipe(tabpfn_anvil_classification_yaml)
    anvil_workflow = anvil_spec.to_workflow()
    anvil_workflow.run(output_dir=tmp_path / "tst")

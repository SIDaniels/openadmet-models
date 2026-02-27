import numpy as np
import pandas as pd
import pytest
import yaml

from openadmet.models.anvil.specification import (
    AnvilSpecification,
    DataSpec,
    EnsembleSpec,
    EvalSpec,
    FeatureSpec,
    Metadata,
    ModelSpec,
    ProcedureSpec,
    ReportSpec,
    SplitSpec,
    TrainerSpec,
)
from openadmet.models.anvil.workflow import AnvilDeepLearningWorkflow, AnvilWorkflow
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


def _build_code_first_anvil_spec(workflow_type: str) -> AnvilSpecification:
    """Build an Anvil specification directly from Python objects."""
    metadata = Metadata(
        version="v1",
        driver="pytorch" if workflow_type == "lightning" else "sklearn",
        name=f"code-first-{workflow_type}",
        build_number=0,
        description="Code-first test workflow",
        tag=f"code-first-{workflow_type}",
        authors="Openadmet tests",
        email="tests@openadmet.org",
        biotargets=["CYP3A4"],
        tags=["openadmet", "unit-test"],
    )
    data = DataSpec(
        type="csv",
        resource="unused.csv",
        input_col="smiles",
        target_cols=["target"],
    )
    procedure = ProcedureSpec(
        split=SplitSpec(
            type="ShuffleSplitter",
            params={
                "train_size": 0.7 if workflow_type == "lightning" else 0.8,
                "val_size": 0.2 if workflow_type == "lightning" else 0.0,
                "test_size": 0.1 if workflow_type == "lightning" else 0.2,
                "random_state": 42,
            },
        ),
        feat=FeatureSpec(
            type="ChemPropFeaturizer"
            if workflow_type == "lightning"
            else "FingerprintFeaturizer",
            params={} if workflow_type == "lightning" else {"fp_type": "ecfp:4"},
        ),
        model=ModelSpec(
            type="ChemPropModel"
            if workflow_type == "lightning"
            else "LGBMRegressorModel",
            params={},
        ),
        train=TrainerSpec(
            type="LightningTrainer"
            if workflow_type == "lightning"
            else "SKLearnBasicTrainer",
            params={
                "max_epochs": 1,
                "accelerator": "cpu",
                "use_wandb": False,
            }
            if workflow_type == "lightning"
            else {},
        ),
    )
    report = ReportSpec(eval=[EvalSpec(type="RegressionMetrics")])
    return AnvilSpecification(
        metadata=metadata,
        data=data,
        procedure=procedure,
        report=report,
    )


@pytest.mark.parametrize("workflow_type", ["sklearn", "lightning"])
def test_anvil_spec_to_workflow_code_first_constructs_expected_workflow(workflow_type):
    """Test code-first workflow construction produces the expected workflow type."""
    anvil_spec = _build_code_first_anvil_spec(workflow_type)
    anvil_workflow = anvil_spec.to_workflow()

    if workflow_type == "lightning":
        assert isinstance(anvil_workflow, AnvilDeepLearningWorkflow)
    else:
        assert isinstance(anvil_workflow, AnvilWorkflow)


@pytest.mark.parametrize("workflow_type", ["sklearn", "lightning"])
def test_anvil_workflow_run_code_first_checks_runtime_seams(
    tmp_path, workflow_type, mocker
):
    """Test code-first run behavior at split and evaluation/report seams."""
    # Build a minimal code-first workflow and synthetic split payloads so this
    # test can focus on orchestration contracts instead of recipe parsing.
    anvil_spec = _build_code_first_anvil_spec(workflow_type)
    anvil_workflow = anvil_spec.to_workflow()
    X = pd.Series(["CCO", "CCN"], name="smiles")
    y = pd.DataFrame({"target": [1.0, 2.0]})
    X_train = pd.Series(["CCO"], name="smiles")
    X_val = pd.Series(["CCC"], name="smiles") if workflow_type == "lightning" else None
    X_test = pd.Series(["CCN"], name="smiles")
    y_train = pd.DataFrame({"target": [1.0]})
    y_val = pd.DataFrame({"target": [1.5]}) if workflow_type == "lightning" else None
    y_test = pd.DataFrame({"target": [2.0]})
    output_dir = tmp_path / f"code_first_{workflow_type}"
    run_tag = "code-first-run-tag"

    # Mock runtime seams that would otherwise perform I/O, featurization, model
    # persistence, and evaluation side effects.
    train_spy = mocker.patch.object(anvil_workflow, "_train")
    read_spy = mocker.patch.object(
        type(anvil_workflow.data_spec),
        "read",
        return_value=(X, y),
    )
    split_spy = mocker.patch.object(
        type(anvil_workflow.split),
        "split",
        return_value=(X_train, X_val, X_test, y_train, y_val, y_test, None),
    )
    featurize_spy = mocker.patch.object(
        type(anvil_workflow.feat),
        "featurize",
        return_value=("mock_loader", None, None, object())
        if workflow_type == "lightning"
        else (np.array([[0.1], [0.2]]), None),
    )
    model_cls = type(anvil_workflow.model)
    serialize_spy = mocker.patch.object(model_cls, "serialize")
    predict_spy = mocker.patch.object(
        model_cls, "predict", return_value=np.array([2.0])
    )
    evaluate_spy = mocker.patch.object(type(anvil_workflow.evals[0]), "evaluate")
    report_spy = mocker.patch.object(type(anvil_workflow.evals[0]), "report")
    if workflow_type == "lightning":
        save_spy = mocker.patch("openadmet.models.anvil.workflow.torch.save")
    else:
        save_spy = mocker.patch("openadmet.models.anvil.workflow.zarr.save")

    # Execute the workflow with mocked seams to validate control-flow behavior.
    anvil_workflow.run(output_dir=output_dir, tag=run_tag)

    # Confirm orchestration hits the expected runtime seams and call counts.
    train_spy.assert_called_once()
    read_spy.assert_called_once()
    split_spy.assert_called_once_with(X, y)
    serialize_spy.assert_called_once()
    predict_spy.assert_called_once()
    evaluate_spy.assert_called_once()
    report_spy.assert_called_once_with(write=True, output_dir=output_dir)
    assert featurize_spy.call_count == 3
    assert save_spy.call_count == (3 if workflow_type == "lightning" else 2)

    # Validate the evaluation payload includes provenance and the held-out target frame.
    evaluate_kwargs = evaluate_spy.call_args.kwargs
    assert evaluate_kwargs["tag"] == run_tag
    assert evaluate_kwargs["target_labels"] == ["target"]
    assert evaluate_kwargs["y_true"].equals(y_test)


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
    assert anvil_workflow.model_kwargs["param_path"] is None
    assert anvil_workflow.model_kwargs["serial_path"] is None
    assert anvil_workflow.ensemble_kwargs == {}
    assert anvil_workflow.feat_kwargs["type"] == anvil_spec.procedure.feat.type


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
    anvil_spec = AnvilSpecification.from_recipe(anvil_full_recipie)
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
    feat_spy = mocker.patch.object(
        type(anvil_workflow.feat),
        "featurize",
        return_value=(np.array([[0.1], [0.2]]), None),
        autospec=True,
    )
    mocker.patch.object(type(anvil_workflow.model), "serialize")
    mocker.patch("openadmet.models.anvil.workflow.zarr.save")
    anvil_spec.run(output_dir=tmp_path / "tst")
    train_spy.assert_called_once()
    assert feat_spy.call_count == 2
    assert (tmp_path / "tst" / "anvil_recipe.yaml").exists()
    assert (tmp_path / "tst" / "recipe_components" / "metadata.yaml").exists()


def test_anvil_spec_run_tag_override_updates_provenance(tmp_path, mocker):
    """Test that a tag override is reflected in the saved provenance recipe."""
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    requested_output_dir = tmp_path / "requested_output"
    resolved_output_dir = tmp_path / "resolved_output"
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    mock_workflow = mocker.Mock()
    mock_workflow.resolved_output_dir = resolved_output_dir
    mocker.patch.object(AnvilSpecification, "to_workflow", return_value=mock_workflow)

    anvil_spec.run(output_dir=requested_output_dir, tag="override-tag")
    mock_workflow.run.assert_called_once_with(
        output_dir=requested_output_dir,
        debug=False,
        tag="override-tag",
    )

    with open(resolved_output_dir / "anvil_recipe.yaml") as stream:
        recipe = yaml.safe_load(stream)
    with open(resolved_output_dir / "recipe_components" / "metadata.yaml") as stream:
        metadata = yaml.safe_load(stream)

    assert recipe["metadata"]["tag"] == "override-tag"
    assert metadata["tag"] == "override-tag"
    assert anvil_spec.metadata.tag != "override-tag"


def test_anvil_spec_run_writes_provenance_to_resolved_output_dir(tmp_path, mocker):
    """Test that provenance is written to the workflow-resolved output directory."""
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    requested_output_dir = tmp_path / "requested_output"
    resolved_output_dir = tmp_path / "resolved_output"
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    mock_workflow = mocker.Mock()
    mock_workflow.resolved_output_dir = resolved_output_dir
    mocker.patch.object(AnvilSpecification, "to_workflow", return_value=mock_workflow)

    anvil_spec.run(output_dir=requested_output_dir)
    mock_workflow.run.assert_called_once_with(
        output_dir=requested_output_dir,
        debug=False,
        tag=None,
    )

    assert (resolved_output_dir / "anvil_recipe.yaml").exists()
    assert (resolved_output_dir / "recipe_components" / "metadata.yaml").exists()
    assert not (requested_output_dir / "anvil_recipe.yaml").exists()


def test_anvil_spec_run_writes_provenance_to_requested_dir_when_no_resolved_output(
    tmp_path, mocker
):
    """Test that provenance falls back to the requested output directory when unresolved."""
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    requested_output_dir = tmp_path / "requested_output"
    assert not requested_output_dir.exists()

    mock_workflow = mocker.Mock()
    mock_workflow.resolved_output_dir = None
    mocker.patch.object(AnvilSpecification, "to_workflow", return_value=mock_workflow)

    anvil_spec.run(
        output_dir=requested_output_dir,
        debug=True,
        tag="fallback-tag",
    )
    mock_workflow.run.assert_called_once_with(
        output_dir=requested_output_dir,
        debug=True,
        tag="fallback-tag",
    )

    assert (requested_output_dir / "anvil_recipe.yaml").exists()
    assert (requested_output_dir / "recipe_components" / "metadata.yaml").exists()


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
    feat_spy = mocker.patch.object(
        type(anvil_workflow.feat),
        "featurize",
        return_value=(np.array([[0.1], [0.2]]), None),
        autospec=True,
    )
    mocker.patch.object(type(anvil_workflow.model), "serialize")

    mocker.patch("openadmet.models.anvil.workflow.zarr.save")
    anvil_workflow.run(output_dir=tmp_path / "tst")
    train_spy.assert_called_once()
    assert feat_spy.call_count == 2


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
    feat_spy = mocker.patch.object(
        type(anvil_workflow.feat),
        "featurize",
        return_value=(np.array([[0.1], [0.2]]), None),
        autospec=True,
    )
    mocker.patch.object(type(anvil_workflow.model), "serialize")

    mocker.patch("openadmet.models.anvil.workflow.zarr.save")
    anvil_workflow.run(output_dir=tmp_path / "tst")
    train_spy.assert_called_once()
    assert feat_spy.call_count == 2


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
    feat_spy = mocker.patch.object(
        type(anvil_workflow.feat),
        "featurize",
        return_value=(object(), None, None, [0]),
        autospec=True,
    )
    mocker.patch.object(type(anvil_workflow.model), "serialize")

    mocker.patch("openadmet.models.anvil.workflow.torch.save")
    anvil_workflow.run(output_dir=tmp_path / "tst")
    train_spy.assert_called_once()
    assert feat_spy.call_count == 1


def test_anvil_workflow_two_way_split_includes_full_dataset_featurization(
    tmp_path, mocker
):
    """
    Test Anvil workflow with a two-way split plus full-dataset featurization.

    Verifies featurization count when train and test sets are present
    and the workflow also featurizes the full dataset for downstream usage.
    """
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    anvil_workflow = anvil_spec.to_workflow()
    X = pd.DataFrame({"smiles": ["CCO", "CCN"]})
    y = pd.DataFrame({"target": [1.0, 2.0]})

    train_spy = mocker.patch.object(type(anvil_workflow), "_train", autospec=True)
    mocker.patch.object(type(anvil_workflow.data_spec), "read", return_value=(X, y))

    # Mock split returning train and test only.
    mocker.patch.object(
        type(anvil_workflow.split),
        "split",
        return_value=(X, None, X, y, None, y, None),
    )

    feat_spy = mocker.patch.object(
        type(anvil_workflow.feat),
        "featurize",
        return_value=(np.array([[0.1], [0.2]]), None),
        autospec=True,
    )
    mocker.patch.object(type(anvil_workflow.model), "serialize")
    mocker.patch.object(
        type(anvil_workflow.model), "predict", return_value=np.array([1.0, 2.0])
    )
    save_spy = mocker.patch("openadmet.models.anvil.workflow.zarr.save")

    anvil_workflow.run(output_dir=tmp_path / "tst")

    train_spy.assert_called_once()
    assert feat_spy.call_count == 3
    assert save_spy.call_count == 2


def test_anvil_workflow_ensemble_bootstrapping(tmp_path, mocker):
    """
    Test Anvil workflow ensemble bootstrapping with a lightweight real model type.

    This test intentionally uses a real sklearn-backed model type
    (DummyRegressorModel) so each bootstrap member behaves like an independent
    model object rather than a pure mock. The goal is to validate ensemble
    orchestration contracts while keeping runtime low.
    """
    anvil_spec = _build_code_first_anvil_spec("sklearn")
    anvil_spec.procedure.model = ModelSpec(
        type="DummyRegressorModel",
        params={"strategy": "mean"},
    )
    anvil_spec.procedure.ensemble = EnsembleSpec(
        type="CommitteeRegressor",
        n_models=3,
        calibration_method="isotonic-regression",
    )
    anvil_spec.procedure.split.params.update(
        {"train_size": 0.7, "val_size": 0.1, "test_size": 0.2}
    )

    anvil_workflow = anvil_spec.to_workflow()

    X = pd.Series(["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCI"], name="smiles")
    y = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    X_train, X_val, X_test = X.iloc[:4], X.iloc[4:5], X.iloc[5:]
    y_train, y_val, y_test = y.iloc[:4], y.iloc[4:5], y.iloc[5:]

    # Runtime seams keep this test fast and deterministic.
    # We keep data ingress and split seams mocked so the workflow control flow
    # is exercised without filesystem or random split variability.
    mocker.patch.object(type(anvil_workflow.data_spec), "read", return_value=(X, y))
    mocker.patch.object(
        type(anvil_workflow.split),
        "split",
        return_value=(X_train, X_val, X_test, y_train, y_val, y_test, None),
    )

    # This seam bypasses expensive chemistry featurization while preserving the
    # invariant that train, val, test, and all-data pathways each consume their
    # own feature matrices.
    train_feat = np.array([[0.0], [1.0], [2.0], [3.0]])
    val_feat = np.array([[4.0]])
    test_feat = np.array([[5.0]])
    full_feat = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    feat_spy = mocker.patch.object(
        type(anvil_workflow.feat),
        "featurize",
        side_effect=[
            (train_feat, None),
            (val_feat, None),
            (test_feat, None),
            (full_feat, None),
        ],
        autospec=True,
    )

    # These seams remove heavyweight persistence and evaluation behavior.
    # They are intentionally narrow: we preserve ensemble construction and
    # bootstrap training behavior while avoiding irrelevant I/O cost.
    mocker.patch("openadmet.models.anvil.workflow.zarr.save")
    mocker.patch.object(type(anvil_workflow.evals[0]), "evaluate")
    mocker.patch.object(type(anvil_workflow.evals[0]), "report")
    serialize_spy = mocker.patch.object(
        type(anvil_workflow.ensemble), "serialize", autospec=True
    )

    bootstrap_indices = [
        np.array([0, 1, 1, 2]),
        np.array([3, 2, 2, 1]),
        np.array([1, 0, 3, 3]),
    ]
    random_choice_spy = mocker.patch(
        "openadmet.models.anvil.workflow.np.random.choice",
        side_effect=bootstrap_indices,
    )
    train_spy = mocker.spy(type(anvil_workflow.trainer), "train")
    calibrate_spy = mocker.patch.object(
        type(anvil_workflow.ensemble),
        "calibrate_uncertainty",
        autospec=True,
    )
    predict_spy = mocker.patch.object(
        type(anvil_workflow.ensemble),
        "predict",
        autospec=True,
        return_value=(np.array([[1.5]]), np.array([[0.2]])),
    )

    anvil_workflow.run(output_dir=tmp_path / "tst")

    assert feat_spy.call_count == 4
    assert len(anvil_workflow.model.models) == anvil_spec.procedure.ensemble.n_models
    bootstrap_models = anvil_workflow.model.models
    assert len({id(model) for model in bootstrap_models}) == len(bootstrap_models)
    assert train_spy.call_count == anvil_spec.procedure.ensemble.n_models

    bootstrap_train_inputs = [call.args[1] for call in train_spy.call_args_list]
    assert len({tuple(arr.reshape(-1)) for arr in bootstrap_train_inputs}) > 1

    calibrate_spy.assert_called_once()
    np.testing.assert_array_equal(calibrate_spy.call_args.args[1], val_feat)
    assert calibrate_spy.call_args.args[2].equals(y_val)
    assert calibrate_spy.call_args.kwargs["method"] == ("isotonic-regression")
    serialize_spy.assert_called_once()
    serialized_ensemble = serialize_spy.call_args.args[0]
    assert hasattr(serialized_ensemble, "models")
    assert len(serialized_ensemble.models) == anvil_spec.procedure.ensemble.n_models
    assert (
        len(serialize_spy.call_args.args[1]) == anvil_spec.procedure.ensemble.n_models
    )
    assert (
        len(serialize_spy.call_args.args[2]) == anvil_spec.procedure.ensemble.n_models
    )

    assert random_choice_spy.call_count == anvil_spec.procedure.ensemble.n_models
    for call in random_choice_spy.call_args_list:
        assert call.kwargs["replace"] is True
        assert call.kwargs["size"] == len(X_train)

    predict_spy.assert_called_once()
    assert predict_spy.call_args.kwargs["return_std"] is True


@pytest.mark.skip(reason="TabPFN requires GPU and is not supported on MacOS runners")
def test_anvil_tabpfn_classification(tmp_path):
    """Test TabPFN classification workflow (skipped on non-GPU environments)."""
    anvil_spec = AnvilSpecification.from_recipe(tabpfn_anvil_classification_yaml)
    anvil_workflow = anvil_spec.to_workflow()
    anvil_workflow.run(output_dir=tmp_path / "tst")

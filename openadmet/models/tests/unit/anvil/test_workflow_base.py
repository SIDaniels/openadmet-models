import pytest
from pydantic import ConfigDict

from openadmet.models.active_learning.ensemble_base import EnsembleBase
from openadmet.models.anvil.specification import DataSpec, Metadata
from openadmet.models.anvil.workflow_base import AnvilWorkflowBase
from openadmet.models.architecture.model_base import PickleableModelBase
from openadmet.models.drivers import DriverType
from openadmet.models.eval.eval_base import EvalBase
from openadmet.models.features.feature_base import FeaturizerBase
from openadmet.models.split.split_base import SplitterBase
from openadmet.models.trainer.trainer_base import TrainerBase


# Concrete workflow implementation for testing
class ConcreteWorkflow(AnvilWorkflowBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def run(self, output_dir="anvil_training", debug=False):
        return "ran"


# Minimal metadata for testing
def get_minimal_metadata():
    return Metadata(
        version="v1",
        driver="sklearn",
        name="test",
        build_number=0,
        description="desc",
        tag="tag",
        authors="auth",
        email="a@b.com",
        biotargets=[],
        tags=[],
    )


# Helper to build a workflow with specific components
def build_workflow(
    mocker,
    *,
    model=None,
    trainer=None,
    evals=None,
    ensemble=None,
    target_cols=["target"],
):
    if model is None:
        model = mocker.create_autospec(PickleableModelBase, instance=True)
        model._n_tasks = 1
        model.n_tasks = 1
        model._driver_type = DriverType.SKLEARN
    if trainer is None:
        trainer = mocker.create_autospec(TrainerBase, instance=True)
        trainer._driver_type = DriverType.SKLEARN
    if evals is None:
        eval_mock = mocker.create_autospec(EvalBase, instance=True)
        eval_mock.is_cross_val = False
        eval_mock._driver_type = DriverType.SKLEARN
        evals = [eval_mock]
    split = mocker.create_autospec(SplitterBase, instance=True)
    split.train_size = 0.8
    split.val_size = 0.0
    split.test_size = 0.2
    feat = mocker.create_autospec(FeaturizerBase, instance=True)
    return ConcreteWorkflow(
        metadata=get_minimal_metadata(),
        data_spec=DataSpec(
            type="csv",
            input_col="smiles",
            target_cols=target_cols,
            resource="data.csv",
        ),
        split=split,
        feat=feat,
        model=model,
        trainer=trainer,
        evals=evals,
        ensemble=ensemble,
    )


# --- Tests ---


def test_multitask_check_passes_when_counts_match(mocker):
    """Test that validation passes when model n_tasks matches data target_cols."""
    model = mocker.create_autospec(PickleableModelBase, instance=True)
    model._n_tasks = 2
    model.n_tasks = 2
    model._driver_type = DriverType.SKLEARN
    workflow = build_workflow(mocker, model=model, target_cols=["t1", "t2"])
    assert workflow


def test_multitask_check_raises_when_counts_mismatch(mocker):
    """Test that validation raises ValueError when n_tasks does not match target_cols."""
    model = mocker.create_autospec(PickleableModelBase, instance=True)
    model._n_tasks = 2
    model.n_tasks = 2
    model._driver_type = DriverType.SKLEARN
    with pytest.raises(ValueError, match="tasks but the data specification has"):
        build_workflow(mocker, model=model, target_cols=["t1", "t2", "t3"])


def test_no_ensemble_cross_val_raises_when_both_present(mocker):
    """Test that using ensemble with cross-validation raises ValueError."""
    ensemble = mocker.create_autospec(EnsembleBase, instance=True)
    eval_mock = mocker.create_autospec(EvalBase, instance=True)
    eval_mock.is_cross_val = True
    eval_mock._driver_type = DriverType.SKLEARN
    with pytest.raises(
        ValueError, match="Ensemble models cannot be used with cross-validation"
    ):
        build_workflow(mocker, ensemble=ensemble, evals=[eval_mock])


def test_no_ensemble_cross_val_allows_cv_without_ensemble(mocker):
    """Test that cross-validation is allowed if no ensemble is present."""
    eval_mock = mocker.create_autospec(EvalBase, instance=True)
    eval_mock.is_cross_val = True
    eval_mock._driver_type = DriverType.SKLEARN
    workflow = build_workflow(mocker, evals=[eval_mock], ensemble=None)
    assert workflow


def test_model_trainer_driver_mismatch_raises(mocker):
    """Test that mismatched model and trainer drivers raise ValueError."""
    model = mocker.create_autospec(PickleableModelBase, instance=True)
    model._n_tasks = 1
    model.n_tasks = 1
    model._driver_type = DriverType.SKLEARN
    trainer = mocker.create_autospec(TrainerBase, instance=True)
    trainer._driver_type = DriverType.LIGHTNING
    with pytest.raises(ValueError, match="Model driver type .* does not match trainer"):
        build_workflow(mocker, model=model, trainer=trainer)


def test_model_trainer_driver_match_succeeds(mocker):
    """Test that matching model and trainer drivers succeed."""
    model = mocker.create_autospec(PickleableModelBase, instance=True)
    model._n_tasks = 1
    model.n_tasks = 1
    model._driver_type = DriverType.SKLEARN
    trainer = mocker.create_autospec(TrainerBase, instance=True)
    trainer._driver_type = DriverType.SKLEARN
    workflow = build_workflow(mocker, model=model, trainer=trainer)
    assert workflow


def test_cv_trainer_compatibility_raises_on_driver_mismatch(mocker):
    """Test that CV evaluator with mismatched trainer driver raises ValueError."""
    trainer = mocker.create_autospec(TrainerBase, instance=True)
    trainer._driver_type = DriverType.SKLEARN
    eval_mock = mocker.create_autospec(EvalBase, instance=True)
    eval_mock.is_cross_val = True
    eval_mock._driver_type = DriverType.LIGHTNING
    with pytest.raises(
        ValueError, match="Trainer driver type .* does not match evaluation"
    ):
        build_workflow(mocker, trainer=trainer, evals=[eval_mock])


def test_cv_trainer_compatibility_ignores_non_cv_evals(mocker):
    """Test that non-CV evaluators do not trigger driver mismatch checks."""
    trainer = mocker.create_autospec(TrainerBase, instance=True)
    trainer._driver_type = DriverType.SKLEARN
    eval_mock = mocker.create_autospec(EvalBase, instance=True)
    eval_mock.is_cross_val = False
    eval_mock._driver_type = DriverType.LIGHTNING
    workflow = build_workflow(mocker, trainer=trainer, evals=[eval_mock])
    assert workflow

import pytest
from pathlib import Path
from typing import Any
from pydantic import ConfigDict
from openadmet.models.anvil.workflow_base import AnvilWorkflowBase
from openadmet.models.anvil.specification import DataSpec, Metadata
from openadmet.models.architecture.model_base import PickleableModelBase
from openadmet.models.trainer.trainer_base import TrainerBase
from openadmet.models.eval.eval_base import EvalBase
from openadmet.models.split.split_base import SplitterBase
from openadmet.models.features.feature_base import FeaturizerBase
from openadmet.models.active_learning.ensemble_base import EnsembleBase


# --- Stub Classes for Testing ---

class ModelStub(PickleableModelBase):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    n_tasks: int = 1
    driver_type: str = "sklearn"

    @property
    def _n_tasks(self):
        return self.n_tasks

    @property
    def _driver_type(self):
        return self.driver_type

    def build(self): pass
    def train(self, X, y): pass
    def predict(self, X, **kwargs): return None


class TrainerStub(TrainerBase):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    driver_type: str = "sklearn"

    @property
    def _driver_type(self):
        return self.driver_type

    def build(self, **kwargs): pass
    def train(self, X=None, y=None): return None


class EvalStub(EvalBase):
    is_cross_val: bool = False
    driver_type: str = "sklearn"

    @property
    def _driver_type(self):
        return self.driver_type

    def evaluate(self, **kwargs): pass
    def report(self, **kwargs): pass


class SplitterStub(SplitterBase):
    def split(self, X, y):
        return (X, None, None, y, None, None, None)


class FeaturizerStub(FeaturizerBase):
    def featurize(self, smiles, *args, **kwargs):
        return (smiles, None)


class EnsembleStub(EnsembleBase):
    def train(self, X, y): pass
    def predict(self, X, **kwargs): return None
    def serialize(self, *args): pass
    def save(self, path): pass
    def load(self, path): pass
    def deserialize(self, *args): pass


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
    model=None,
    trainer=None,
    evals=None,
    ensemble=None,
    target_cols=["target"],
):
    return ConcreteWorkflow(
        metadata=get_minimal_metadata(),
        data_spec=DataSpec(
            type="csv",
            input_col="smiles",
            target_cols=target_cols,
            resource="data.csv",
        ),
        split=SplitterStub(),
        feat=FeaturizerStub(),
        model=model or ModelStub(),
        trainer=trainer or TrainerStub(),
        evals=evals or [EvalStub()],
        ensemble=ensemble,
    )


# --- Tests ---

def test_multitask_check_passes_when_counts_match():
    """Test that validation passes when model n_tasks matches data target_cols."""
    # 2 tasks, 2 target cols
    model = ModelStub(n_tasks=2)
    workflow = build_workflow(model=model, target_cols=["t1", "t2"])
    assert workflow


def test_multitask_check_raises_when_counts_mismatch():
    """Test that validation raises ValueError when n_tasks does not match target_cols."""
    # 2 tasks, 3 target cols
    model = ModelStub(n_tasks=2)
    with pytest.raises(ValueError, match="tasks but the data specification has"):
        build_workflow(model=model, target_cols=["t1", "t2", "t3"])


def test_no_ensemble_cross_val_raises_when_both_present():
    """Test that using ensemble with cross-validation raises ValueError."""
    ensemble = EnsembleStub()
    evals = [EvalStub(is_cross_val=True)]
    
    with pytest.raises(ValueError, match="Ensemble models cannot be used with cross-validation"):
        build_workflow(ensemble=ensemble, evals=evals)


def test_no_ensemble_cross_val_allows_cv_without_ensemble():
    """Test that cross-validation is allowed if no ensemble is present."""
    evals = [EvalStub(is_cross_val=True)]
    workflow = build_workflow(evals=evals, ensemble=None)
    assert workflow


def test_model_trainer_driver_mismatch_raises():
    """Test that mismatched model and trainer drivers raise ValueError."""
    model = ModelStub(driver_type="sklearn")
    trainer = TrainerStub(driver_type="lightning")

    with pytest.raises(ValueError, match="Model driver type .* does not match trainer"):
        build_workflow(model=model, trainer=trainer)


def test_model_trainer_driver_match_succeeds():
    """Test that matching model and trainer drivers succeed."""
    model = ModelStub(driver_type="sklearn")
    trainer = TrainerStub(driver_type="sklearn")
    workflow = build_workflow(model=model, trainer=trainer)
    assert workflow


def test_cv_trainer_compatibility_raises_on_driver_mismatch():
    """Test that CV evaluator with mismatched trainer driver raises ValueError."""
    trainer = TrainerStub(driver_type="sklearn")
    evals = [EvalStub(is_cross_val=True, driver_type="lightning")]

    with pytest.raises(ValueError, match="Trainer driver type .* does not match evaluation"):
        build_workflow(trainer=trainer, evals=evals)


def test_cv_trainer_compatibility_ignores_non_cv_evals():
    """Test that non-CV evaluators do not trigger driver mismatch checks."""
    trainer = TrainerStub(driver_type="sklearn")
    # Even if eval driver is different, if is_cross_val is False, it should pass
    evals = [EvalStub(is_cross_val=False, driver_type="lightning")]

    workflow = build_workflow(trainer=trainer, evals=evals)
    assert workflow

import pytest
import pandas as pd
import numpy as np
from typing import Any
from unittest.mock import MagicMock
from pathlib import Path
from pydantic import ConfigDict

from openadmet.models.anvil.workflow import (
    AnvilWorkflow,
    AnvilDeepLearningWorkflow,
    _safe_to_numpy,
    _DRIVER_TO_CLASS,
)
from openadmet.models.drivers import DriverType
from openadmet.models.anvil.specification import DataSpec, Metadata
from openadmet.models.architecture.model_base import PickleableModelBase, LightningModelBase
from openadmet.models.trainer.trainer_base import TrainerBase
from openadmet.models.eval.eval_base import EvalBase
from openadmet.models.split.split_base import SplitterBase
from openadmet.models.features.feature_base import FeaturizerBase
from openadmet.models.active_learning.ensemble_base import EnsembleBase
from openadmet.models.transforms.transform_base import TransformBase


# --- Pydantic Stub Classes ---

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
    accelerator: str = "cpu"
    devices: int = 1
    use_wandb: bool = False
    output_dir: Any = None
    driver_type: str = "sklearn"

    @property
    def _driver_type(self):
        return self.driver_type

    def build(self, **kwargs): pass
    def train(self, X=None, y=None): return None


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


class TransformStub(TransformBase):
    def transform(self, X, *args, **kwargs): return X


class DLModelStub(LightningModelBase):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    n_tasks: int = 1

    @property
    def _n_tasks(self):
        return self.n_tasks

    def build(self, **kwargs): pass
    def train(self, *args, **kwargs): pass
    def predict(self, X, **kwargs): return None


class DLTrainerStub(TrainerBase):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    accelerator: str = "cpu"
    devices: int = 1
    use_wandb: bool = False
    output_dir: Any = None

    @property
    def _driver_type(self):
        return DriverType.LIGHTNING

    def build(self, **kwargs): pass
    def train(self, train_dl=None, val_dl=None): return None


class DLFeaturizerStub(FeaturizerBase):
    def featurize(self, smiles, *args, **kwargs):
        return (MagicMock(), None, None, MagicMock())


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

def make_workflow(cls, **kwargs):
    model = kwargs.pop("model", ModelStub())
    trainer = kwargs.pop("trainer", TrainerStub(driver_type=model.driver_type))
    split = kwargs.pop("split", SplitterStub())
    feat = kwargs.pop("feat", FeaturizerStub())
    ensemble = kwargs.pop("ensemble", None)
    transform = kwargs.pop("transform", None)

    defaults = {
        "metadata": get_minimal_metadata(),
        "data_spec": DataSpec(
            type="csv", input_col="smiles", target_cols=["target"], resource="data.csv"
        ),
        "split": split,
        "feat": feat,
        "model": model,
        "trainer": trainer,
        "evals": [],
        "ensemble": ensemble,
        "transform": transform,
        "model_kwargs": {},
        "ensemble_kwargs": {},
        "feat_kwargs": {},
    }
    defaults.update(kwargs)

    wf = cls(**defaults)

    # Attach method mocks after construction so tests can assert on them
    object.__setattr__(wf.model, "build", MagicMock())
    object.__setattr__(wf.model, "make_new", MagicMock(return_value=wf.model))
    object.__setattr__(wf.model, "serialize", MagicMock())
    object.__setattr__(wf.model, "predict", MagicMock(return_value=np.array([1.0])))
    train_mock = MagicMock(return_value=wf.model)
    object.__setattr__(wf.trainer, "train", train_mock)
    object.__setattr__(wf.trainer, "build", MagicMock())
    if wf.ensemble is not None:
        object.__setattr__(wf.ensemble, "from_models", MagicMock(return_value=wf.model))

    return wf


def make_dl_workflow(**kwargs):
    model = kwargs.pop("model", DLModelStub())
    trainer = kwargs.pop("trainer", DLTrainerStub())
    split = kwargs.pop("split", SplitterStub())
    feat = kwargs.pop("feat", DLFeaturizerStub())
    ensemble = kwargs.pop("ensemble", None)

    defaults = {
        "metadata": get_minimal_metadata(),
        "data_spec": DataSpec(
            type="csv", input_col="smiles", target_cols=["target"], resource="data.csv"
        ),
        "split": split,
        "feat": feat,
        "model": model,
        "trainer": trainer,
        "evals": [],
        "ensemble": ensemble,
        "model_kwargs": {},
        "ensemble_kwargs": {},
        "feat_kwargs": {},
    }
    defaults.update(kwargs)

    wf = AnvilDeepLearningWorkflow(**defaults)

    object.__setattr__(wf.model, "build", MagicMock())
    object.__setattr__(wf.model, "deserialize", MagicMock(return_value=wf.model))
    object.__setattr__(wf.model, "serialize", MagicMock())
    object.__setattr__(wf.model, "predict", MagicMock(return_value=np.array([1.0])))
    train_mock = MagicMock(return_value=wf.model)
    object.__setattr__(wf.trainer, "train", train_mock)
    object.__setattr__(wf.trainer, "build", MagicMock())

    return wf


# --- Unit Tests ---

def test_safe_to_numpy_converts_series():
    """Test _safe_to_numpy converts pd.Series to np.ndarray."""
    s = pd.Series([1.0, 2.0, 3.0])
    res = _safe_to_numpy(s)
    assert isinstance(res, np.ndarray)
    assert res.shape == (3,)
    assert np.allclose(res, [1.0, 2.0, 3.0])


def test_safe_to_numpy_converts_dataframe():
    """Test _safe_to_numpy converts pd.DataFrame to np.ndarray."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    res = _safe_to_numpy(df)
    assert isinstance(res, np.ndarray)
    assert res.shape == (2, 2)


def test_safe_to_numpy_passthrough_numpy_array():
    """Test _safe_to_numpy passes through np.ndarray."""
    arr = np.array([1.0, 2.0])
    res = _safe_to_numpy(arr)
    assert res is arr


def test_anvilworkflow_check_if_val_needed_raises_for_ensemble_without_val():
    """Test validation raises if ensemble is used without validation set."""
    with pytest.raises(ValueError, match="Ensemble models require a validation set"):
        make_workflow(
            AnvilWorkflow,
            ensemble=EnsembleStub(),
            split=SplitterStub(train_size=1.0, val_size=0.0, test_size=0.0),
            ensemble_kwargs={"n_models": 2},
        )


def test_anvilworkflow_check_no_finetuning_raises_with_model_path():
    """Test validation raises if finetuning paths are provided for single model."""
    with pytest.raises(ValueError, match="Finetuning .* is not supported"):
        make_workflow(
            AnvilWorkflow,
            model_kwargs={"param_path": "p.pt"}
        )


def test_anvilworkflow_check_no_finetuning_raises_with_ensemble_path():
    """Test validation raises if finetuning paths are provided for ensemble."""
    with pytest.raises(ValueError, match="Finetuning .* is not supported"):
        make_workflow(
            AnvilWorkflow,
            ensemble=EnsembleStub(),
            ensemble_kwargs={"param_paths": ["p.pt"]},
            split=SplitterStub(train_size=0.8, val_size=0.1, test_size=0.1),
        )


def test_anvildeeplearningworkflow_check_no_transform_raises():
    """Test DL workflow raises if transform is provided."""
    with pytest.raises(ValueError, match="Transform step is not supported"):
        make_workflow(
            AnvilDeepLearningWorkflow,
            transform=TransformStub(),
            trainer=TrainerStub(driver_type="lightning"),
            model=ModelStub(n_tasks=1, driver_type="lightning"),
        )


def test_anvildeeplearningworkflow_check_if_val_needed_raises_for_ensemble_without_val():
    """Test DL workflow raises if ensemble is used without validation set."""
    with pytest.raises(ValueError, match="Ensemble models require a validation set"):
        make_workflow(
            AnvilDeepLearningWorkflow,
            ensemble=EnsembleStub(),
            split=SplitterStub(train_size=1.0, val_size=0.0, test_size=0.0),
            trainer=TrainerStub(driver_type="lightning"),
            model=ModelStub(n_tasks=1, driver_type="lightning"),
        )


def test_anvilworkflow_train_calls_build_and_train(tmp_path, mocker):
    """Test _train method calls model.build and trainer.train, and updates workflow.model."""
    workflow = make_workflow(AnvilWorkflow)
    
    X_train = pd.Series(["C", "CC"])
    y_train = pd.DataFrame({"target": [1.0, 2.0]})
    
    # Capture original model before _train updates workflow.model
    original_model = workflow.model
    sentinel_model = ModelStub()
    workflow.trainer.train.return_value = sentinel_model
    
    workflow._train(X_train, y_train, tmp_path)
    
    original_model.build.assert_called_once()
    workflow.trainer.train.assert_called_once()
    assert workflow.model is sentinel_model


def test_anvilworkflow_train_ensemble_calls_trainer_n_models_times(tmp_path, mocker):
    """Test _train_ensemble calls trainer n_models times."""
    workflow = make_workflow(
        AnvilWorkflow,
        ensemble=EnsembleStub(),
        ensemble_kwargs={"n_models": 3},
        split=SplitterStub(train_size=0.8, val_size=0.1, test_size=0.1),
    )
    
    X_train = np.array([[1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 2.0, 3.0])
    
    # Mock make_new to return self or new stub
    workflow.model.make_new.return_value = workflow.model
    workflow.trainer.train.return_value = workflow.model
    
    workflow._train_ensemble(X_train, y_train, tmp_path)
    
    assert workflow.trainer.train.call_count == 3
    assert workflow.model.build.call_count == 3
    workflow.ensemble.from_models.assert_called_once()


def test_anvilworkflow_run_without_test_skips_eval(tmp_path, mocker):
    """Test run() skips evaluation if no test set is produced."""
    X_train = pd.Series(["C"])
    y_train = pd.DataFrame({"target": [1]})

    workflow = make_workflow(AnvilWorkflow)
    mocker.patch.object(DataSpec, "read", autospec=True, return_value=(X_train, y_train))
    mocker.patch.object(SplitterStub, "split", autospec=True,
                        return_value=(X_train, None, None, y_train, None, None, None))
    mocker.patch.object(FeaturizerStub, "featurize", autospec=True,
                        return_value=(np.array([[1]]), None))
    mocker.patch("openadmet.models.anvil.workflow.zarr.save")

    eval_mock = MagicMock()
    workflow.evals = [eval_mock]

    workflow.run(output_dir=tmp_path)

    eval_mock.evaluate.assert_not_called()


def test_anvilworkflow_run_with_test_calls_eval(tmp_path, mocker):
    """Test run() calls evaluation when test set is present."""
    X_train = pd.Series(["C"])
    y_train = pd.DataFrame({"target": [1.0]})
    X_test = pd.Series(["CC"])
    y_test = pd.DataFrame({"target": [2.0]})
    X = pd.Series(["C", "CC"])
    y = pd.DataFrame({"target": [1.0, 2.0]})

    workflow = make_workflow(AnvilWorkflow)
    mocker.patch.object(DataSpec, "read", autospec=True, return_value=(X, y))
    mocker.patch.object(SplitterStub, "split", autospec=True,
                        return_value=(X_train, None, X_test, y_train, None, y_test, None))
    mocker.patch.object(FeaturizerStub, "featurize", autospec=True,
                        return_value=(np.array([[1.0]]), None))
    mocker.patch("openadmet.models.anvil.workflow.zarr.save")

    # Mock model prediction
    workflow.model.predict.return_value = np.array([2.0])

    eval_mock = MagicMock()
    workflow.evals = [eval_mock]

    workflow.run(output_dir=tmp_path)

    eval_mock.evaluate.assert_called_once()
    eval_mock.report.assert_called_once()

    call_kwargs = eval_mock.evaluate.call_args.kwargs
    assert call_kwargs["tag"] == "tag"
    assert call_kwargs["target_labels"] == ["target"]
    assert call_kwargs["y_true"].shape == (1, 1)
    assert call_kwargs["y_true"].iloc[0, 0] == pytest.approx(2.0)


def test_anvilworkflow_run_classification_uses_predict_proba(tmp_path, mocker):
    """Test run() uses predict_proba for classification if available."""
    X_train = pd.Series(["C"])
    y_train = pd.DataFrame({"target": [0]})
    X_test = pd.Series(["CC"])
    y_test = pd.DataFrame({"target": [1]})
    X = pd.Series(["C", "CC"])
    y = pd.DataFrame({"target": [0, 1]})

    workflow = make_workflow(AnvilWorkflow)
    mocker.patch.object(DataSpec, "read", autospec=True, return_value=(X, y))
    mocker.patch.object(SplitterStub, "split", autospec=True,
                        return_value=(X_train, None, X_test, y_train, None, y_test, None))
    mocker.patch.object(FeaturizerStub, "featurize", autospec=True, return_value=(np.array([[1]]), None))
    mocker.patch("openadmet.models.anvil.workflow.zarr.save")

    # Attach predict_proba mock to model instance
    object.__setattr__(workflow.model, "predict_proba",
                       MagicMock(return_value=np.array([[0.1, 0.9]])))

    workflow.run(output_dir=tmp_path)

    workflow.model.predict_proba.assert_called_once()


def test_driver_to_class_mapping():
    """Test driver to class mapping dictionary."""
    assert _DRIVER_TO_CLASS[DriverType.SKLEARN] == AnvilWorkflow
    assert _DRIVER_TO_CLASS[DriverType.LIGHTNING] == AnvilDeepLearningWorkflow


# --- AnvilDeepLearningWorkflow tests (Refinement 7) ---

def test_anvildeeplearningworkflow_train_single_model_build_from_scratch(tmp_path):
    """Test DL _train builds model from scratch and updates workflow.model."""
    workflow = make_dl_workflow()

    # Capture original model before _train updates workflow.model
    original_model = workflow.model
    sentinel_model = DLModelStub()
    workflow.trainer.train.return_value = sentinel_model

    workflow._train(None, None, None, tmp_path)

    original_model.build.assert_called_once()
    workflow.trainer.train.assert_called_once()
    assert workflow.model is sentinel_model


def test_anvildeeplearningworkflow_train_deserializes_when_paths_provided(tmp_path):
    """Test DL _train deserializes model when param_path and serial_path are provided."""
    workflow = make_dl_workflow(model_kwargs={"param_path": "p.pt", "serial_path": "s.pt"})

    sentinel_model = DLModelStub()
    original_model = workflow.model
    original_model.deserialize.return_value = sentinel_model
    workflow.trainer.train.return_value = sentinel_model

    workflow._train(None, None, None, tmp_path)

    original_model.deserialize.assert_called_once()
    original_model.build.assert_not_called()
    workflow.trainer.train.assert_called_once()


def test_anvildeeplearningworkflow_run_single_model(tmp_path, mocker):
    """Test AnvilDeepLearningWorkflow run completes for single model with no test set."""
    X_data = pd.Series(["C"], name="smiles")
    y_data = pd.DataFrame({"target": [1.0]})

    workflow = make_dl_workflow()

    mock_dl = MagicMock()
    mock_ds = MagicMock()
    mocker.patch.object(DataSpec, "read", autospec=True, return_value=(X_data, y_data))
    mocker.patch.object(SplitterStub, "split", autospec=True,
                        return_value=(X_data, None, None, y_data, None, None, None))
    mocker.patch.object(DLFeaturizerStub, "featurize", autospec=True,
                        return_value=(mock_dl, None, None, mock_ds))
    mocker.patch("openadmet.models.anvil.workflow.torch.save")

    workflow.run(output_dir=tmp_path / "out")

    workflow.trainer.train.assert_called_once()
    assert workflow.resolved_output_dir is not None

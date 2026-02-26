from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openadmet.models.inference import inference as inference_module


@pytest.fixture
def input_df():
    return pd.DataFrame({"MY_SMILES": ["CCO", "CCN"]})


def test_predict_with_mocked_single_model(mocker, input_df):
    mock_model = mocker.Mock()
    mock_model.estimator = "mock-estimator"
    mock_model.predict.return_value = np.asarray([[1.0], [2.0]])
    mock_feat = mocker.Mock()
    mock_feat.featurize.return_value = (np.asarray([[0.1], [0.2]]), np.array([0, 1]))
    mock_metadata = mocker.Mock()
    mock_metadata.tag = "UNIT"
    mock_data_spec = mocker.Mock()
    mock_data_spec.target_cols = ["task_0"]

    mock_loader = mocker.patch.object(
        inference_module,
        "load_anvil_model_and_metadata",
        return_value=(mock_model, mock_feat, mock_metadata, mock_data_spec),
    )

    result = inference_module.predict(
        input_path=input_df,
        input_col="MY_SMILES",
        model_dir=["unused-model-dir"],
        accelerator="cpu",
        log=False,
    )

    assert isinstance(result, pd.DataFrame)
    assert "OADMET_PRED_UNIT_task_0" in result.columns
    assert "OADMET_STD_UNIT_task_0" in result.columns
    assert result["OADMET_PRED_UNIT_task_0"].tolist() == [1.0, 2.0]
    assert result["OADMET_STD_UNIT_task_0"].isna().all()
    mock_loader.assert_called_once_with(Path("unused-model-dir"))


def test_predict_with_mocked_ensemble_and_acquisition(mocker, input_df):
    mock_model = mocker.Mock()
    mock_model.estimator = "mock-ensemble"
    mock_model.n_models = 2
    mock_model.predict.return_value = (
        np.asarray([[0.6], [0.4]]),
        np.asarray([[0.05], [0.15]]),
    )
    mock_feat = mocker.Mock()
    mock_feat.featurize.return_value = (np.asarray([[0.1], [0.2]]), np.array([0, 1]))
    mock_metadata = mocker.Mock()
    mock_metadata.tag = "ENS"
    mock_data_spec = mocker.Mock()
    mock_data_spec.target_cols = ["task_0"]

    mocker.patch.object(inference_module, "EnsembleBase", type(mock_model))
    mock_loader = mocker.patch.object(
        inference_module,
        "load_anvil_model_and_metadata",
        return_value=(mock_model, mock_feat, mock_metadata, mock_data_spec),
    )

    result = inference_module.predict(
        input_path=input_df,
        input_col="MY_SMILES",
        model_dir=["unused-model-dir"],
        accelerator="cpu",
        log=False,
        aq_fxn_args={"ucb": {"beta": 2.0}},
    )

    pred_values = result["OADMET_PRED_ENS_task_0"].tolist()
    std_values = result["OADMET_STD_ENS_task_0"].tolist()
    ucb_values = result["OADMET_UCB_ENS_task_0"].tolist()

    assert pred_values == pytest.approx([0.6, 0.4])
    assert std_values == pytest.approx([0.05, 0.15])
    assert ucb_values == pytest.approx([0.7, 0.7])
    mock_loader.assert_called_once_with(Path("unused-model-dir"))


def test_predict_raises_when_input_column_missing(input_df):
    with pytest.raises(ValueError, match="Column OTHER not found"):
        inference_module.predict(
            input_path=input_df,
            input_col="OTHER",
            model_dir=["unused-model-dir"],
            log=False,
        )


def test_load_anvil_model_and_metadata_missing_recipe_components(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not contain recipe components"):
        inference_module.load_anvil_model_and_metadata(tmp_path)


def test_load_anvil_model_and_metadata_missing_procedure_yaml(tmp_path):
    model_dir = tmp_path / "model"
    recipe_components = model_dir / "recipe_components"
    recipe_components.mkdir(parents=True)
    (recipe_components / "metadata.yaml").write_text("metadata")
    (recipe_components / "data.yaml").write_text("data")

    with pytest.raises(FileNotFoundError, match="does not contain procedure.yaml"):
        inference_module.load_anvil_model_and_metadata(model_dir)

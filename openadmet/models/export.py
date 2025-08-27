import onnx
import torch


def export_to_onnx_from_pytorch(model, dataloader, output_path, dynamo=True):
    model.eval()
    with torch.no_grad():
        for datum in dataloader:
            print(datum)
            _export_pytorch_model_to_onnx(model, datum, output_path, dynamo=dynamo)
            break  # Export only the first batch


def _export_pytorch_model_to_onnx(model, input_tensor, output_path, dynamo=True):
    # Export the PyTorch model to ONNX format
    torch.onnx.export(model, input_tensor, output_path, dynamo=dynamo, export_params=True)

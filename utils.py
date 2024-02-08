"""
Contains various utility functions for PyTorch model saving
"""
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def create_writer(experiment_name: str, model_name: str, extra: str = None):
    from datetime import datetime
    import os

    timestamp = datetime.now().strftime("%Y-%m-%d")
    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}......")
    return SummaryWriter(log_dir=log_dir)


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """
    Saves a PyTorch model to a target directory.

    Args:
      model: A target PyTorch model to save.
      target_dir: A directory for saving the model to.
      model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.
    """

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name


    # save onnx model
    save_onnx_model(model_name=model_name, model=model, target_dir=target_dir)


    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def save_onnx_model(model_name:str, model:torch.nn.Module, target_dir):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("[INFO] Saving model in onnx format")
    model_name = model_name.split(".")[0] + ".onnx"
    torch_input = torch.randn(1, 3, 224, 224, device=device)
    onnx_program = torch.onnx.dynamo_export(model, torch_input)
    save_path = target_dir +"/" + model_name
    onnx_program.save(save_path)
    print(f"[INFO] Sucessfully saved model in onnx format in {save_path}")

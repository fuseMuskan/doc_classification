"""
Trains a PyTorch image classification model using device-agnostic code
"""
import argparse
import torch
from pathlib import Path
import model_builder
from ultralytics import YOLO

import engine
from utils import create_writer, save_model
import custom_focal_loss


# Extracting argparse values

parser = argparse.ArgumentParser()

parser.add_argument("--MODEL", type=str, help="Model name [Alexnet, Resnet,...]")
parser.add_argument("--EPOCHS", type=int, help="Number of epochs for training")
parser.add_argument("--BATCH_SIZE", type=int, help="Batch size for training")
parser.add_argument("--LEARNING_RATE", type=float, help="Learning rate")
parser.add_argument("--DATA_DIR", type=str, help="Path to the data directory")
parser.add_argument(
    "--USE_CLASS_WEIGHTS",
    type=bool,
    help="if True Uses Focal Loss Function with class weights",
)
parser.add_argument("--OUTPUT_MODEL", type=str, help="Name of the model to be saved")


args = parser.parse_args()

# Set up Hyperparameters
MODEL_NAME = args.MODEL
EPOCHS = args.EPOCHS
BATCH_SIZE = args.BATCH_SIZE
LEARNING_RATE = args.LEARNING_RATE
DATA_DIR = args.DATA_DIR
USE_CLASS_WEIGHTS = args.USE_CLASS_WEIGHTS
OUTPUT_MODEL = args.OUTPUT_MODEL


# Setup directories
data_path = Path(DATA_DIR)
train_dir = data_path / "train"
test_dir = data_path / "test"
val_dir = data_path / "validation"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

if MODEL_NAME == "yolo":
  print("[INFO] Loading YOLO V8 Pretrained Model")
  model = YOLO('yolov8n-cls.pt')
  print("[INFO] Loaded YOLO V8 Pretrained Model")
  model.train(data=data_path, epochs=EPOCHS)

(
    model,
    input_size,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    class_names,
) = model_builder.initialize_model(
    model_name=MODEL_NAME,
    num_classes=4,
    feature_extract=True,
    train_dir=train_dir,
    test_dir=test_dir,
    val_dir=val_dir,
    batch_size=BATCH_SIZE,
)


# Set loss and optimizer
# if USE_CLASS_WEIGHTS:
#     # uses class weights with focal losss function
#     print("Creating.. Focal Loss Function with class weights")
#     loss_fn = custom_focal_loss.create_focal_loss_criterion(
#         train_dataloader, class_names, gamma=2
#     )
#     print("Created Focal Loss Function with class weights")
# else:
    # Uses default loss function
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Create summary writer to track experiment
writer = create_writer("doc_classification", MODEL_NAME, f"{EPOCHS} epochs")


# Start training with help from engine.py
engine.train(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=EPOCHS,
    writer=writer,
    device=device,
    model_name=MODEL_NAME,
)

# Save the model with help from utils.py
save_model(model=model, target_dir="models", model_name=f"{OUTPUT_MODEL}.pth")
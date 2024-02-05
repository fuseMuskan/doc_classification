import torch
from torch import nn
from eval import (
    calculate_confusion_matrix,
    save_confusion_matrix,
    compute_classification_report,
)

"""
Contains functions for training and testing a Pytorch model
"""


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    confusion_matrix = torch.zeros((4, 4), dtype=torch.float32, device=device)

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):

        print(device)
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        confusion_matrix += calculate_confusion_matrix(y_pred_class, y)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc, confusion_matrix


def val_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    val_loss, val_acc = 0, 0

    confusion_matrix = torch.zeros((4, 4), dtype=torch.float32)

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            val_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            # Calculate and accumulate accuracy
            val_pred_labels = val_pred_logits.argmax(dim=1)
            val_acc += (val_pred_labels == y).sum().item() / len(val_pred_labels)
            confusion_matrix += calculate_confusion_matrix(val_pred_labels, y)

    # Adjust metrics to get average loss and accuracy per batch
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc, confusion_matrix


from tqdm.auto import tqdm

# 1. Take in various parameters required for training and test steps
def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    writer,
    model_name: str,
    loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
    track_experiment: bool = True,
):

    # 2. Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_conf_matrix = train_step(
            model=model,
            device=device,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        val_loss, val_acc, val_conf_matrix = val_step(
            model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=device
        )

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        # # Print and log confusion matrices
        # print("Train Confusion Matrix:")
        # print(train_conf_matrix)
        # print("Test Confusion Matrix:")
        # print(test_conf_matrix)

        # generate images of confusion marix
        train_conf_fig = save_confusion_matrix(train_conf_matrix)
        val_conf_fig = save_confusion_matrix(val_conf_matrix)

        if track_experiment:
            # Experiment tracking
            # Add loss results to SummaryWriter
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "val_loss": val_loss},
                global_step=epoch,
            )

            # Add confusion matrix image to SummaryWriter
            writer.add_figure("train_conf_fig", train_conf_fig, global_step=epoch)
            writer.add_figure("val_conf_fig", val_conf_fig, global_step=epoch)

            # Add accuracy results to SummaryWriter
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={"train_acc": train_acc, "val_acc": val_acc},
                global_step=epoch,
            )

            # Track the PyTorch model architecture
            writer.add_graph(
                model=model,
                # Pass in an example input
                input_to_model=torch.randn(1, 3, 224, 224).to(device),
            )

            # Close the writer
            writer.close()
        else:
            pass

    final_model = model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            output = final_model(X)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.argmax(output, dim=1).cpu().numpy())

    # Use the compute_classification_report function
    compute_classification_report(
        torch.tensor(y_pred), torch.tensor(y_true), model_name=model_name
    )

    # 6. Return the filled results at the end of the epochs
    return results

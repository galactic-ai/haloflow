import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .utils import EarlyStopper


def train_dann(
    model, 
    train_loader, 
    test_loader, 
    num_epochs=50, 
    lr=0.001, 
    patience=5,  # Early stopping patience
    min_delta=0.001,  # Minimum change in loss to qualify as an improvement
    device="cuda"
):
    # Loss functions
    criterion_task = nn.MSELoss()  # For stellar/halo mass prediction (regression)
    criterion_domain = nn.CrossEntropyLoss()  # For domain classification

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Early stopping
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    # Move model to device
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch, domain_batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}"
        ):
            X_batch, y_batch, domain_batch = (
                X_batch.to(device),
                y_batch.to(device),
                domain_batch.to(device),
            )

            # Forward pass
            label_pred, domain_pred = model(X_batch)

            # Compute losses
            loss_task = criterion_task(label_pred, y_batch)
            loss_domain = criterion_domain(domain_pred, domain_batch)
            loss = loss_task + loss_domain  # Total loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print epoch loss
        avg_loss = total_loss / len(train_loader)

        # Evaluate on test domain (optional)
        ave_loss_test = evaluate(model, test_loader, device)
        if early_stopper.early_stop(ave_loss_test):
            print("Early stopping")
            break
        scheduler.step(ave_loss_test)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Test Loss: {ave_loss_test:.4f}")


def evaluate(model, test_loader, device="cuda"):
    model.eval()
    total_mse = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            label_pred, _ = model(X_batch)
            total_mse += nn.MSELoss()(label_pred, y_batch).item()

    avg_mse = total_mse / len(test_loader)
    return avg_mse

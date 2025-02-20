import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_regression(model, dataloader, device="cuda"):
    """
    Evaluate the model's regression performance (MSE, RMSE, R²).

    Parameters
    ----------
    model : torch.nn.Module
        Trained regression model.
    dataloader : torch.utils.data.DataLoader
        DataLoader for the test set.
    device : str
        Device to run evaluation on.

    Returns
    -------
    dict
        Dictionary containing MSE, RMSE, and R² scores.
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds, _ = model(X_batch)
            y_true.append(y_batch.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return {"mse": mse, "rmse": rmse, "r2": r2}


def domain_accuracy(model, dataloader, device="cuda"):
    """
     Evaluate the domain classifier's accuracy.

    Parameters
    ----------
     model : torch.nn.Module
         Trained domain classifier model.
     dataloader : torch.utils.data.DataLoader
         DataLoader for the test set.
     device : str
         Device to run evaluation on.

     Returns
     -------
     float
         Domain classification accuracy.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, _, domain_batch in dataloader:
            X_batch, domain_batch = X_batch.to(device), domain_batch.to(device)
            _, domain_pred = model(X_batch)
            preds = domain_pred.argmax(dim=1)
            correct += (preds == domain_batch).sum().item()
            total += domain_batch.size(0)

    acc = correct / total
    return acc

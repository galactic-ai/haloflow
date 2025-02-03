import torch
import torch.nn as nn
import torch.optim as optim

from . import model as M

def train(X_source, Y_source, X_target, hidden_dim=64,
          alpha=1.0, num_epochs=100, batch_size=64, lr=1e-3):
    """
    Train the DANN model.

    Parameters
    ----------
    X_source : torch.Tensor
        Source data.
    Y_source : torch.Tensor
        Source labels.
    X_target : torch.Tensor
        Target data.
    hidden_dim : int
        Hidden dimension. Default is 64.
    alpha : float
        Gradient reversal weight. Default is 1.0.
    num_epochs : int
        Number of epochs. Default is 100.
    batch_size : int
        Batch size. Default is 64.
    lr : float
        Learning rate. Default is 1e-3.
    
    Returns
    -------
    model : torch.nn.Module
        Trained model of the DANN.
    """

    # Initialize model, optimizer, and loss functions
    input_dim = X_source.shape[1]  # Number of features
    num_classes = Y_source.shape[1]  # Number of output classes (e.g., stellar and halo mass)

    model = M.DANN(input_dim, hidden_dim, num_classes, alpha)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    label_criterion = nn.MSELoss()  # Use MSE for regression (stellar and halo mass)
    domain_criterion = nn.BCELoss()  # Use BCE for domain classification

    for epoch in range(num_epochs):
        model.train()
        for i in range(0, len(X_source), batch_size):
            # Get batches
            src_X_batch = X_source[i:i+batch_size]
            src_Y_batch = Y_source[i:i+batch_size]
            tgt_X_batch = X_target[i:i+batch_size]

            # Forward pass
            src_label_pred, src_domain_pred = model(src_X_batch, alpha)
            _, tgt_domain_pred = model(tgt_X_batch, alpha)

            # Compute losses
            label_loss = label_criterion(src_label_pred, src_Y_batch)
            domain_loss = domain_criterion(src_domain_pred, torch.ones(src_X_batch.size(0), 1)) + \
                domain_criterion(tgt_domain_pred, torch.zeros(tgt_X_batch.size(0), 1))

            # Total loss
            total_loss = label_loss + domain_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}")
    
    return model


def evaluate(model, X_test, Y_test, alpha=1.):
    """
    Evaluate the DANN model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model of the DANN.
    X_test : torch.Tensor
        Test data.
    Y_test : torch.Tensor
        Test labels.
    alpha : float
        Gradient reversal weight. Default is 1.0.
    
    Returns
    -------
    label_pred : torch.Tensor
        Predicted labels.
    label_loss
        Loss of the predicted labels.
    """

    model.eval()

    label_criterion = nn.MSELoss()  # Use MSE for regression (stellar and halo mass)

    with torch.no_grad():
        label_pred, _ = model(X_test, alpha=alpha)
        # Compute loss
        label_loss = label_criterion(label_pred, Y_test)
        print(f"Test Loss: {label_loss.item():.4f}")
    
    return label_pred, label_loss
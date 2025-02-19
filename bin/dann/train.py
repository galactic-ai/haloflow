import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from haloflow.dann import utils as U
from haloflow.dann import model as M
from haloflow.dann import data_loader as D
from haloflow import config as C

try:
    import wandb
except ImportError:
    wandb = None

def train_dann(config, use_wandb=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sims = ['TNG50', 'TNG100', 'Eagle100', 'Simba100', 'TNG_ALL']

    # Initialize W&B
    if use_wandb and wandb is not None:
        wandb.init(config=config)
        config = wandb.config # Overwrite config with W&B config

    dataset = D.SimulationDataset(
        sims, 
        config["obs"], 
        C.get_dat_dir()
    )
    # choose a test sim not in train_sim
    test_sim = [sim for sim in sims if sim not in config["train_sim"]][0]
    train_loader, test_loader = dataset.get_train_test_loaders(
        config["train_sim"],
        test_sim,
        config["batch_size"],
    )

    # Infer input dimension from data
    sample_X, _, _ = next(iter(train_loader))
    config['input_dim'] = sample_X.shape[1]

    # Model
    model = M.DANN(
        input_dim=config["input_dim"],
        feature_layers=config["feature_layers"],
        label_layers=config["label_layers"],
        domain_layers=config["domain_layers"],
        alpha=config["alpha"],
        num_domains=len(config["train_sim"]),
    ).to(device)

    # Loss functions
    criterion_task = nn.MSELoss()  # For stellar/halo mass prediction (regression)
    criterion_domain = nn.CrossEntropyLoss()  # For domain classification

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Early stopping
    early_stopper = U.EarlyStopper(
        patience=config["es_patience"], 
        min_delta=config["es_min_delta"],
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    for epoch in range(config["num_epochs"]):
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
        loss_test = evaluate(model, test_loader, device)

        # Log to W&B or print locally
        if use_wandb and wandb is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "test_loss": loss_test,
                "lr": optimizer.param_groups[0]["lr"]
            })
        else:
            print(f"Epoch {epoch + 1}/{config['num_epochs']}")
            print(f"Train Loss: {avg_loss:.4f}, Test Loss: {loss_test:.4f}")

        # Early stopping
        if not use_wandb and early_stopper.early_stop(loss_test):
            print("Early stopping")
            break

        # Scheduler step
        scheduler.step(loss_test)
    
    # save model
    if use_wandb and wandb is not None:
        torch.save(model.state_dict(), f'{C.get_dat_dir()}/hf2/dann/models/dann_model_{wandb.run.id}.pt')
    else:
        torch.save(model.state_dict(), f'{C.get_dat_dir()}/hf2/dann/models/dann_model_final.pt')
    


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

if __name__ == "__main__":
    config = {}
    train_dann(config, use_wandb=True)
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from haloflow.dann import utils as U
from haloflow.dann import model as M
from haloflow.dann import data_loader as D
from haloflow.dann import evalutate as E
from haloflow.dann import visualise as V
from haloflow import config as C

try:
    import wandb
except ImportError:
    wandb = None

def train_dann(config, use_wandb=True, plots=True):
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
        # '../../data/'
    )
    # choose a test sim not in train_sim
    test_sim = [sim for sim in sims if sim not in config["train_sim"]][0]
    train_loader, test_loader = dataset.get_train_test_loaders(
        config["train_sim"],
        test_sim,
        config["batch_size"],
    )
    
    # since we will evaluate on stellar and halo mass
    eval_scaler = dataset.scaler_Y 

    # Infer input dimension from data
    sample_X, _, _ = next(iter(train_loader))
    config['input_dim'] = sample_X.shape[1]

    # Model
    model = M.DANN(
        input_dim=config["input_dim"],
        feature_layers=config["feature_layers"],
        label_layers=config["label_layers"],
        domain_layers=config["domain_layers"],
        alpha=0,
        num_domains=len(config["train_sim"]) + 1,
    ).to(device)
    
    if use_wandb and wandb is not None:
        wandb.watch(model, log='all')

    # Loss functions
    criterion_task = nn.MSELoss()  # For stellar/halo mass prediction (regression)
    criterion_domain = nn.CrossEntropyLoss()  # For domain classification

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    # Early stopping
    early_stopper = U.EarlyStopper(
        patience=config["es_patience"], 
        min_delta=config["es_min_delta"],
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=True
    )

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0.0

        i = 0
        for X_batch, y_batch, domain_batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}"
        ):
            p = float(i + epoch * len(train_loader)) / config["num_epochs"] / len(train_loader)
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            i += 1
            model.domain_classifier.grl.update_alpha(alpha)
            X_batch, y_batch, domain_batch = (
                X_batch.to(device),
                y_batch.to(device),
                domain_batch.to(device),
            )

            # Forward pass
            label_pred, domain_pred = model(X_batch)
            # print(label_pred)

            # Compute losses
            loss_task = criterion_task(label_pred, y_batch)
            loss_domain = criterion_domain(domain_pred, domain_batch)
            # Domain loss should be subtracted as the 
            # domain classifier is trying to minimize it
            loss = loss_task + loss_domain  # Total loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f'Loss task: {loss_task.item()}, Loss domain: {loss_domain.item()}')

        # Print epoch loss
        avg_loss = total_loss / len(train_loader)

        # Evaluate on test domain (optional)
        loss_test = evaluate(model, test_loader, eval_scaler, device)
        domain_acc = E.domain_accuracy(model, train_loader, device)
        # Log to W&B or print locally
        if use_wandb and wandb is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "test_loss": loss_test,
                "lr": optimizer.param_groups[0]["lr"],
                "domain_accuracy": domain_acc
            })
        else:
            print(f"Epoch {epoch + 1}/{config['num_epochs']}")
            print(f"Train Loss: {avg_loss:.4f}, Test Loss: {loss_test:.4f}, Domain Accuracy: {domain_acc:.4f}")

        # Early stopping
        if early_stopper.early_stop(loss_test):
            print("Early stopping")
            break

        # Scheduler step
        scheduler.step(loss_test)
        
    if plots:
        emb, labels = V.visualize_features_fast(model, train_loader, test_loader, n_samples=5000, device='cpu')
        fig = V.plot_combined_tsne(emb, labels,)
        if use_wandb and wandb is not None:
            wandb.log({'TNSE_Plot': wandb.Image(fig)})
    
    # save model
    if use_wandb and wandb is not None:
        torch.save(model.state_dict(), f'{C.get_dat_dir()}/hf2/dann/models/dann_model_{wandb.run.id}.pt')
    else:
        # naming model_train_sims_to_test_sim_obs_obs_lrlr_bsbs_eepoch_timestamp.pt
        name = f'dann_model_{"_".join(config["train_sim"])}_to_{test_sim}_{config["obs"]}_lr{config["lr"]}_bs{config["batch_size"]}_e{epoch+1}_{U.get_timestamp()}.pt'
        torch.save(model.state_dict(), f'{C.get_dat_dir()}/hf2/dann/models/{name}')
    


def evaluate(model, test_loader, scaler, device="cuda"):
    model.eval()
    total_mse = 0.0

    with torch.no_grad():
        for X_batch, y_batch, _ in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            label_pred, _ = model(X_batch) 
            total_mse += nn.MSELoss()(label_pred, y_batch).item()

    avg_mse = total_mse / len(test_loader)
    return avg_mse

if __name__ == "__main__":
    config = {}
    train_dann(config, use_wandb=True)

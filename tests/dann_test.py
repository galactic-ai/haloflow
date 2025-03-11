import sys

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import umap
import matplotlib.pyplot as plt

import numpy as np
from haloflow.dann.data_loader import SimulationDataset
from sklearn.metrics import mean_squared_error, r2_score

import tensorboardX

summary_writer = tensorboardX.SummaryWriter("experiment1")

class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANNModel(nn.Module):
    def __init__(self, input_dim, num_domains):
        super(DANNModel, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        
        # # add attention mechanism
        # self.attention = nn.Sequential(
        #     nn.Linear(64, 32),
        #     nn.ReLU(True),
        #     nn.Linear(32, 64),
        #     nn.Softmax(dim=1)
        # )
        
        self.class_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, 2)  # Output layer for two continuous values (halo mass and stellar mass)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, num_domains),  # Output layer for domain classification
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        
        # attention = self.attention(feature)
        # feature = feature * attention
        
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output, feature
        
        

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
    feature_list = []
    domains = []
    correct_domains, total = 0, 0

    with torch.no_grad():
        for X_batch, y_batch, label in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds, domain_output, feature = model(X_batch, alpha=0)
            y_true.append(y_batch.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            feature_list.append(feature.cpu().numpy())
            domains.append(label.cpu().numpy())
            
            # Append domain outputs
            preds = domain_output.argmax(dim=1)
            correct_domains += torch.sum(preds == label).item()
            total += len(label)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    feature_list = np.concatenate(feature_list)
    domains = np.concatenate(domains)
    
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(feature_list)
    
    sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=domains, cmap='viridis')
    plt.colorbar(sc)
    summary_writer.add_figure('UMAP', plt.gcf(), global_step=epoch)
    
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    domain_acc = correct_domains / total

    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Domain Accuracy: {domain_acc:.4f}")
    return {"mse": mse, "rmse": rmse, "r2": r2, "domain_acc": domain_acc}



## Training stuff

source_datasets = ['TNG100', 'TNG50', 'Eagle100']
target_dataset = 'Simba100'
obs = 'mags'

all_sims = ['TNG100', 'TNG50', 'Eagle100', 'Simba100']

data_dir = '../../data/hf2/'

batch_size = 64
lr = 1e-3
num_epochs = 250 
device = 'cpu'

# Dataset
sim_dataset = SimulationDataset(all_sims, obs, data_dir)
train_loader, test_loader = sim_dataset.get_train_test_loaders(source_datasets, target_dataset, batch_size)

X_train, _, _ = next(iter(train_loader))
input_dim = X_train.shape[1]

num_domains = len(source_datasets) + 1

my_net = DANNModel(input_dim, num_domains).to(device)

# optimizer
optimizer = torch.optim.Adam(my_net.parameters(), lr=lr)


# loss functions
loss_class = nn.MSELoss().to(device)
loss_domain = nn.NLLLoss().to(device)

for param in my_net.parameters():
    param.requires_grad = True

# training
best_accu_t = float('inf')
for epoch in range(num_epochs):
    len_dataloader = min(len(train_loader), len(test_loader))
    data_source_iter = iter(train_loader)
    data_target_iter = iter(test_loader)
    
    for i in range(len_dataloader):
        p = float(i + epoch * len_dataloader) / num_epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        data_source = next(data_source_iter)
        X, y, label = data_source
        X = X.to(device)
        y = y.to(device)
        label = label.to(device)
        
        my_net.zero_grad()
        batch_size = len(y)
        
        class_output, domain_output, _ = my_net(X, alpha)
        
        err_s_label = loss_class(class_output, y)
        err_s_domain = loss_domain(domain_output, label)
        
        # target data
        data_target = next(data_target_iter)
        t_X, t_y, t_label = data_target
        t_X = t_X.to(device)
        t_y = t_y.to(device)
        t_label = t_label.to(device)
        
        t_batch_size = len(t_y)
        
        _, t_domain_output, _ = my_net(t_X, alpha)
        
        err_t_domain = loss_domain(t_domain_output, t_label)
        
        err = err_s_label + err_s_domain + err_t_domain
        
        summary_writer.add_scalar('source_loss', err_s_label.item(), epoch)
        summary_writer.add_scalar('domain_loss', err_s_domain.item(), epoch)
        summary_writer.add_scalar('target_loss', err_t_domain.item(), epoch)
        summary_writer.add_scalar('total_loss', err.item(), epoch)
        
        err.backward()
        optimizer.step()
        
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                                % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                                   err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
        sys.stdout.flush() 
        
        torch.save(my_net, '../../data/hf2/dann/models/dann_model_epoch_current.pth')
    
    print('\n')
    print('Evaluation results (target):')
    eval_t = evaluate_regression(my_net, test_loader, device='cpu')
    summary_writer.add_scalar('target_rmse', eval_t['rmse'], epoch)
    summary_writer.add_scalar('target_mse', eval_t['mse'], epoch)
    summary_writer.add_scalar('target_r2', eval_t['r2'], epoch)
    summary_writer.add_scalar('target_domain_acc', eval_t['domain_acc'], epoch)
    
    print('Evaluation results (source):')
    eval_s = evaluate_regression(my_net, train_loader, device='cpu')
    summary_writer.add_scalar('source_rmse', eval_s['rmse'], epoch)
    summary_writer.add_scalar('source_mse', eval_s['mse'], epoch)
    summary_writer.add_scalar('source_r2', eval_s['r2'], epoch)
    summary_writer.add_scalar('source_domain_acc', eval_s['domain_acc'], epoch)
    
    accu_t = eval_t['rmse']
    accu_s = eval_s['rmse']
    
    print('Test accuracy on target dataset:', accu_t)
    print('Train accuracy on source dataset:', accu_s)
    
    if accu_t < best_accu_t:
        best_accu_t = accu_t
        torch.save(my_net, '../../data/hf2/dann/models/dann_model_best.pth')


print('Best accuracy on target dataset:', best_accu_t)
print('Model saved in ../../data/hf2/dann/models/dann_model_best.pth')
print('Training complete.')

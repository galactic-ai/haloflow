import torch

def mmd_loss(x, y, sigma=1.0):
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    X_sq = xx.diag().unsqueeze(1) + xx.diag().unsqueeze(0) - 2 * xx
    Y_sq = yy.diag().unsqueeze(1) + yy.diag().unsqueeze(0) - 2 * yy
    XY_sq = x.pow(2).sum(dim=1).unsqueeze(1) + y.pow(2).sum(dim=1).unsqueeze(0) - 2 * xy

    X_exp = torch.exp(-X_sq / (2 * sigma**2))
    Y_exp = torch.exp(-Y_sq / (2 * sigma**2))
    XY_exp = torch.exp(-XY_sq / (2 * sigma**2))

    loss = (X_exp.mean() + Y_exp.mean() - 2 * XY_exp.mean()) * 0.5
    return loss

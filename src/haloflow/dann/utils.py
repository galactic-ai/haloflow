import torch


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None  # Reverse gradients


def get_timestamp():
    import datetime

    # will return a string like '2021-06-01'
    return datetime.datetime.now().strftime("%Y-%m-%d")

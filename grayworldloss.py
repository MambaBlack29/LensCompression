import torch

def grayworldloss(residual):

    mean_rgb = torch.mean(residual, dim = [2, 3])

    r, g, b = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]

    loss = torch.mean((r-g)**2 + (g-r)**2 + (g-b)**2)

    return loss
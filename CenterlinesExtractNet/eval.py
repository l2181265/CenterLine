import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    centerlines_type = torch.float32 if net.n_classes == 1 else torch.long
    points_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_centerlines, true_points = batch['image'], batch['centerlines'], batch['points']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_centerlines = true_centerlines.to(device=device, dtype=centerlines_type)
            true_points = true_points.to(device=device, dtype=points_type)

            with torch.no_grad():
                centerlines_pred, points_pred = net(imgs)

            if net.n_classes > 1:
                LOSS = F.cross_entropy(centerlines_pred, true_centerlines).item() + F.cross_entropy(points_pred, true_points).item()
                tot += LOSS
            else:
                centerlines_pred = torch.sigmoid(centerlines_pred)
                points_pred = torch.sigmoid(points_pred)
                centerlines_pred = (centerlines_pred > 0.5).float()
                points_pred = (points_pred > 0.5).float()
                LOSS = dice_coeff(centerlines_pred, true_centerlines).item() + dice_coeff(points_pred, true_points).item()
                tot += LOSS
            pbar.update()

    net.train()
    return tot / n_val

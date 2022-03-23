import torch


def vMF3_log_likelihood(y_true, mu_pred, kappa_pred):
    cosin_dist = torch.sum(y_true * mu_pred, dim=1)
    log_likelihood = kappa_pred * cosin_dist + torch.log(kappa_pred) - torch.log(1 - torch.exp(-2*kappa_pred)) - kappa_pred

    return log_likelihood


def compute_vMF3_loss(outputs, true_dir):
    mu = outputs['direction']
    kappa = outputs['kappa']

    if true_dir.shape[-1] == 2:
        mu = mu[..., :2] / torch.norm(mu[..., :2], dim=-1, keepdim=True)

    mu = mu.reshape(-1, mu.shape[-1])
    kappa = kappa.reshape(-1, kappa.shape[-1])
    true_dir = true_dir.reshape(-1, true_dir.shape[-1])
    values = vMF3_log_likelihood(true_dir, mu, kappa)
    return -values.mean()


def compute_kappa_vMF3_loss(outputs, true_dir):
    mu = outputs['direction'].detach()
    kappa = outputs['kappa']

    if true_dir.shape[-1] == 2:
        mu = mu[..., :2] / torch.norm(mu[..., :2], dim=-1, keepdim=True)

    mu = mu.reshape(-1, mu.shape[-1])
    kappa = kappa.reshape(-1, kappa.shape[-1])
    true_dir = true_dir.reshape(-1, true_dir.shape[-1])
    values = vMF3_log_likelihood(true_dir, mu, kappa)
    return -values.mean()


def compute_basic_cos_loss(outputs, true_dir):
    """
    Compute integrated loss function of spherical regression
    """

    reg_dir = outputs['direction']

    reg_dir = reg_dir.reshape(-1, reg_dir.shape[-1])
    true_dir = true_dir.reshape(-1, true_dir.shape[-1])

    if true_dir.shape[-1] == 2:
        reg_dir = reg_dir[..., :2] / torch.norm(reg_dir[..., :2], dim=-1, keepdim=True)

    cos = torch.sum(reg_dir * true_dir, dim=-1)
    cos[cos > 1] = 1
    cos[cos < -1] = -1
    loss = 1 - cos

    return loss.mean()

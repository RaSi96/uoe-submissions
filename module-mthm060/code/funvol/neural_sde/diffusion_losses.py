import numpy as np
import torch

# ------------------------------------------------------------------------------

def log_likelihood_loss(
        nn_mu: torch.Tensor,
        nn_Sigma: torch.Tensor,
        dT: torch.Tensor,
        dX: torch.Tensor
    ) -> torch.Tensor:
    Z = torch.distributions.MultivariateNormal(
        loc               = nn_mu*dT[:, None],
        covariance_matrix = nn_Sigma*dT[:, None, None]
    )

    return -Z.log_prob(dX).mean()


def density_loss(
        nn_mu: torch.Tensor,
        nn_Sigma: torch.Tensor,
        dT: torch.Tensor,
        dX: torch.Tensor
    ) -> torch.Tensor:
    # PIT (this section confirmed identical to FuNVol authors' code)
    Z = torch.distributions.Normal(
        loc   = nn_mu*dT[:, None],
        scale = torch.sqrt( nn_Sigma.diagonal(dim1=-2, dim2=-1)*dT[:, None] )
    )
    pit = Z.cdf(dX)

    # KDE nodes
    u = torch.linspace(0, 1, 101, device=pit.device)
    du = u[1]-u[0]

    # Silverman's rule (same as authors). used for scaling the loss
    h = 0.1*1.069 * pit.detach().std(dim=0) * (pit.shape[0]**(-1/5))

    # PIT: [B, N, 1]
    # u  : [1, 1, G]
    # h  : [1, N, 1]
    z = (
        u[None, None, :]
        - pit[:, :, None]
    ) / h[None, :, None]

    # Gaussian kernel for KDE
    phi = torch.exp(-0.5 * z**2) / np.sqrt(2*np.pi)

    # normalising KDE by bandwidth
    f = (phi / h[None, :, None]).mean(dim=0)

    # quadrature
    return ((f-1.0)**2).sum(dim=1).mul(du).sum()

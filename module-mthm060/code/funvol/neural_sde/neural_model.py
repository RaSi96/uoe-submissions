import torch
import torch.nn as nn

# ------------------------------------------------------------------------------

class _GRU(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_layers: int) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size  = n_in,
            hidden_size = n_hidden,
            num_layers  = n_layers,
            batch_first = True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return h.transpose(0, 1).flatten(1)


class GRU_drift(nn.Module):
    def __init__(
            self,
            n_in: int,
            n_hidden: int,
            n_layers: int,
            n_out: int
        ) -> None:
        super().__init__()
        self.gru = _GRU(n_in, n_hidden, n_layers)
        self.linear = nn.Linear(n_hidden*n_layers, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.gru(x))


class GRU_diffusion(nn.Module):
    def __init__(
            self,
            n_in: int,
            n_hidden: int,
            n_layers: int,
            n_out: int
        ) -> None:
        super().__init__()
        self.n_out = n_out
        self.gru = _GRU(n_in, n_hidden, n_layers)

        # this linear layer outputs Cholesky L. input is GRU final hidden state
        self.linear = nn.Linear(
            n_hidden * n_layers,
            n_out * (n_out + 1) // 2
        )

        with torch.no_grad():
            self.linear.weight.uniform_(-0.01, 0.01)
            self.linear.bias.fill_(1e-3)

        # need these for Mr Chief Guest Cholesky
        self.register_buffer("tril", torch.tril_indices(n_out, n_out))
        self.tril: torch.Tensor

        self.register_buffer("I", torch.eye(n_out))
        self.I: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.gru(x)

        L = x.new_zeros(x.size(0), self.n_out, self.n_out)
        L[:, self.tril[0], self.tril[1]] = self.linear(h)

        L.diagonal(dim1=-2, dim2=-1).exp_()

        # FuNVol authors add 1e-3*self.I, but I feel exponentiating is more
        # natural for keeping something PSD. could be wrong though!
        return L @ L.transpose(-1, -2)  # + 1e-3*self.I


class NeuralSDE(nn.Module):
    def __init__(self, nn_params: dict) -> None:
        super().__init__()
        self.drift = GRU_drift(**nn_params)
        self.diffusion = GRU_diffusion(**nn_params)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Returns conditional mean vector \mu and diffusion matrix \Sigma in a
        tuple (mu, Sigma).
        """
        mu = self.drift(x)
        Sigma = self.diffusion(x)
        return mu, Sigma

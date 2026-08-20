"""Lightweight PyTorch MLP surrogate model for predicting adversarial routing costs."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class SurrogateMLP(nn.Module):
    """Multilayer Perceptron to predict expected adversarial travel ticks from network state."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_surrogate(
    features: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    *,
    epochs: int = 150,
    lr: float = 0.01,
    batch_size: int = 8,
    hidden_dim: int = 32,
    device: str = "cpu",
) -> tuple[SurrogateMLP, list[float]]:
    """Train the SurrogateMLP on the collected SBO dataset.

    Parameters
    ----------
    features:
        Input tensor or array of shape (num_samples, input_dim).
        Contains depot indicator and demands.
    targets:
        Target tensor or array of shape (num_samples,) or (num_samples, 1).
        Contains expected adversarial travel times/ticks.
    epochs:
        Number of epochs to train for.
    lr:
        Learning rate for Adam optimizer.
    batch_size:
        Mini-batch size for DataLoader.
    hidden_dim:
        Hidden layer dimension size.
    device:
        Torch device to execute training on (e.g. 'cpu', 'mps').

    Returns
    -------
    Tuple of (trained_model, list_of_epoch_losses).
    """
    if isinstance(features, np.ndarray):
        features = torch.tensor(features, dtype=torch.float32)
    if isinstance(targets, np.ndarray):
        targets = torch.tensor(targets, dtype=torch.float32)

    if targets.dim() == 1:
        targets = targets.unsqueeze(1)

    input_dim = features.shape[1]
    model = SurrogateMLP(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    generator = torch.Generator()
    generator.manual_seed(42)
    dataset = TensorDataset(features.to(device), targets.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    epoch_losses = []
    model.train()
    for epoch in range(epochs):
        batch_losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        epoch_loss = float(np.mean(batch_losses))
        epoch_losses.append(epoch_loss)

    return model, epoch_losses

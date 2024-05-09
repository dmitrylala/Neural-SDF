import argparse
from pathlib import Path
from typing import List

import numpy as np

from common import load_points
from layers import SirenNetwork, MSELoss, getNetwork, Adam


def batchify(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False):
    batch_idxs = np.arange((X.shape[0] + batch_size - 1) // batch_size)

    if shuffle:
        np.random.shuffle(batch_idxs)

    for batch_idx in batch_idxs:
        idx = slice(batch_idx * batch_size, (batch_idx + 1) * batch_size)
        yield X[idx, :], y[idx]


def fit(
    net: SirenNetwork,
    optim,
    n_epochs: int,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    log_every_n_epochs: int = -1,
) -> List[float]:
    loss_logs = []
    loss = MSELoss()
    for epoch in range(n_epochs):
        epoch_losses = []
        for x_batch, y_batch in batchify(X, y, batch_size, shuffle=True):
            preds = net(x_batch.T)
            loss_value = loss(preds, y_batch)
            net.backward(loss.backward())
            optim.step()

            epoch_losses.append(loss_value)

        if log_every_n_epochs > 0 and epoch % log_every_n_epochs == 0:
            print(f"Epoch = {epoch}, loss = {np.mean(epoch_losses)}")
        loss_logs.append(loss_value)
    return loss_logs


def train(
    train_points: Path,
    n_hidden: int,
    hidden_size: int,
    n_epochs: int,
    lr: float,
    batch_size: int,
    save_to: Path,
    verbose: bool = False,
) -> None:
    net = getNetwork(n_hidden, hidden_size)
    X_train, y_train = load_points(train_points)
    opt = Adam(net.params(), lr=lr)
    _ = fit(
        net,
        opt,
        n_epochs,
        X_train,
        y_train,
        batch_size,
        log_every_n_epochs=100 if verbose else -1,
    )
    net.save(save_to)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SIREN network with numpy")
    parser.add_argument(
        "--train_points", required=True, type=Path, help="Path to train points .bin"
    )
    parser.add_argument(
        "--n_hidden", type=int, required=True, help="Number of hidden layers"
    )
    parser.add_argument(
        "--hidden_size", type=int, required=True, help="Hidden layers size"
    )
    parser.add_argument("--n_epochs", type=int, default=1500, help="Train for n_epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Adam learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--save_to", type=Path, required=True, help="Where to save weights"
    )
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()

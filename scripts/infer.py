import argparse
import struct
from pathlib import Path

import numpy as np

W0 = 30
INPUT_DIM = 3
OUT_DIM = 1


def load_points(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("i", f.read(4))
        points = np.fromfile(f, dtype=np.float32, count=3 * n, sep="").reshape(-1, 3)
        sdfs = np.fromfile(f, dtype=np.float32, count=n, sep="")
    return points, sdfs


class Linear:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def __call__(self, x):
        return self.w @ x + self.b.reshape(self.b.shape[0], -1)

    def n_params(self):
        return np.prod(self.w.shape) + np.prod(self.b.shape)


class SinActivation:
    def __init__(self, w0):
        self.w0 = w0

    def __call__(self, x):
        return np.sin(self.w0 * x)

    def n_params(self):
        return 0


class Network:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def n_params(self):
        return sum(layer.n_params() for layer in self.layers)


def load_siren_network(
    path: str, n_hidden: int, hidden_size: int, in_dim=INPUT_DIM, out_dim=OUT_DIM
):
    layers = []
    with open(path, "rb") as f:
        w = np.fromfile(
            f, dtype=np.float32, count=in_dim * hidden_size, sep=""
        ).reshape(hidden_size, in_dim)
        b = np.fromfile(f, dtype=np.float32, count=hidden_size, sep="")
        layers.extend([Linear(w, b), SinActivation(W0)])

        for _ in range(n_hidden):
            w = np.fromfile(
                f, dtype=np.float32, count=hidden_size**2, sep=""
            ).reshape(hidden_size, hidden_size)
            b = np.fromfile(f, dtype=np.float32, count=hidden_size, sep="")
            layers.extend([Linear(w, b), SinActivation(W0)])

        w = np.fromfile(
            f, dtype=np.float32, count=hidden_size * out_dim, sep=""
        ).reshape(out_dim, hidden_size)
        b = np.fromfile(f, dtype=np.float32, count=out_dim, sep="")
        layers.append(Linear(w, b))

    return Network(*layers)


def compare_sdfs(
    points_path: Path,
    weights_path: Path,
    n_hidden: int,
    hidden_size: int,
    n_points_show: int = 10,
) -> None:
    net = load_siren_network(weights_path, n_hidden=n_hidden, hidden_size=hidden_size)
    print(f"Loaded network, total params: {net.n_params()}")

    points, gt_sdfs = load_points(points_path)
    print(f"Loaded points, shape: {points.shape}")

    print(f"Comparing first {n_points_show} sdf values with predictions:")
    print(gt_sdfs[:n_points_show], net(points[:n_points_show, :].T).flatten())


def main() -> None:
    parser = argparse.ArgumentParser(description="Load weights and infer SIREN network")
    parser.add_argument("weights_path", type=Path, help="Path to weights.bin")
    parser.add_argument("points_path", type=Path, help="Path to points.bin")
    parser.add_argument(
        "--n_hidden", type=int, required=True, help="Number of hidden layers"
    )
    parser.add_argument(
        "--hidden_size", type=int, required=True, help="Hidden layers size"
    )
    parser.add_argument(
        "--n_points_show", type=int, default=10, help="Number of first points to infer"
    )
    args = parser.parse_args()
    compare_sdfs(**vars(args))


if __name__ == "__main__":
    main()

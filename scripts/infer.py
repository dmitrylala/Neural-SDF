import argparse
from pathlib import Path

import numpy as np

from common import load_points
from layers import Linear, SinActivation, SirenNetwork


def load_siren_network(path: str, n_hidden: int, hidden_size: int):
    in_dim, out_dim = 3, 1

    layers = []
    with open(path, "rb") as f:
        w = np.fromfile(
            f, dtype=np.float32, count=in_dim * hidden_size, sep=""
        ).reshape(hidden_size, in_dim)
        b = np.fromfile(f, dtype=np.float32, count=hidden_size, sep="").reshape(
            hidden_size, 1
        )
        linear = Linear(in_dim, hidden_size)
        linear.w = w
        linear.b = b
        layers.extend([linear, SinActivation()])

        for _ in range(n_hidden):
            w = np.fromfile(
                f, dtype=np.float32, count=hidden_size**2, sep=""
            ).reshape(hidden_size, hidden_size)
            b = np.fromfile(f, dtype=np.float32, count=hidden_size, sep="").reshape(
                hidden_size, 1
            )
            linear = Linear(hidden_size, hidden_size)
            linear.w = w
            linear.b = b
            layers.extend([linear, SinActivation()])

        w = np.fromfile(
            f, dtype=np.float32, count=hidden_size * out_dim, sep=""
        ).reshape(out_dim, hidden_size)
        b = np.fromfile(f, dtype=np.float32, count=out_dim, sep="").reshape(out_dim, 1)
        linear = Linear(hidden_size, hidden_size)
        linear.w = w
        linear.b = b
        layers.append(linear)

    return SirenNetwork(*layers)


def compare_sdfs(
    points: Path,
    weights: Path,
    n_hidden: int,
    hidden_size: int,
    n_points_show: int = 10,
) -> None:
    net = load_siren_network(weights, n_hidden=n_hidden, hidden_size=hidden_size)
    print(f"Loaded network, total params: {net.n_params()}")

    pts, gt_sdfs = load_points(points)
    print(f"Loaded points, shape: {pts.shape}")

    print(f"Comparing first {n_points_show} sdf values with predictions:")
    print(gt_sdfs[:n_points_show], net(pts[:n_points_show, :].T).flatten())


def main() -> None:
    parser = argparse.ArgumentParser(description="Load weights and infer SIREN network")
    parser.add_argument(
        "--weights", required=True, type=Path, help="Path to weights.bin"
    )
    parser.add_argument("--points", required=True, type=Path, help="Path to points.bin")
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

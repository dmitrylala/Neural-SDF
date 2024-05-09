from typing import Tuple
import struct
import numpy as np


def load_points(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("i", f.read(4))
        points = np.fromfile(f, dtype=np.float32, count=3 * n, sep="").reshape(-1, 3)
        sdfs = np.fromfile(f, dtype=np.float32, count=n, sep="")
    return points, sdfs

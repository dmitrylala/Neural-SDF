from typing import List
import struct

import numpy as np

W0 = 30.0


class Parameter(np.ndarray):
    def __new__(cls, arr):
        obj = np.asarray(arr).view(cls)
        obj.grad = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.grad = getattr(obj, "grad", None)


class Linear:
    def __init__(self, in_dim: int, out_dim: int):
        rng = np.random.default_rng()
        c = np.sqrt(6 / in_dim) / W0
        self.w = Parameter(rng.uniform(-c, c, size=(out_dim, in_dim)))
        self.b = Parameter(np.zeros((out_dim, 1)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        out = np.array(self.w @ x + self.b)
        return out

    def backward(self, dLdy: np.ndarray) -> np.ndarray:
        self.w.grad = dLdy @ self.x.T
        self.b.grad = np.sum(dLdy, axis=1, keepdims=True)
        dLdX = self.w.T @ dLdy
        return dLdX

    def params(self) -> List[Parameter]:
        return [self.w, self.b]

    def n_params(self) -> int:
        return sum(np.prod(p.shape) for p in self.params())


class SinActivation:
    def __init__(self, w0: int = W0):
        self.w0 = w0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.sin(self.w0 * x)

    def backward(self, dLdy: np.ndarray) -> np.ndarray:
        dLdx = self.w0 * np.cos(self.w0 * self.x) * dLdy
        return dLdx

    def params(self) -> List[Parameter]:
        return []

    def n_params(self) -> int:
        return 0


class SirenNetwork:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, dLdy: np.ndarray) -> np.ndarray:
        for layer in self.layers[::-1]:
            dLdy = layer.backward(dLdy)

    def params(self) -> List[Parameter]:
        layers_params = []
        for layer in self.layers:
            layers_params.extend(layer.params())
        return layers_params

    def save(self, save_to: str) -> None:
        params = list(np.concatenate([p.flatten() for p in self.params()]))
        s = struct.pack("f" * len(params), *params)
        with open(save_to, "wb") as f:
            f.write(s)

    def n_params(self) -> int:
        return sum(layer.n_params() for layer in self.layers)


def getNetwork(n_hidden: int, hidden_size: int) -> SirenNetwork:
    in_dim, out_dim = 3, 1

    layers = []
    layers.extend([Linear(in_dim, hidden_size), SinActivation()])
    for _ in range(n_hidden):
        layers.extend([Linear(hidden_size, hidden_size), SinActivation()])
    layers.append(Linear(hidden_size, out_dim))

    return SirenNetwork(*layers)


class MSELoss:
    def __call__(self, X: np.ndarray, y: np.ndarray):
        self.saved = X, y
        loss = (X - y) ** 2
        return loss.mean()

    def backward(self) -> np.ndarray:
        X, y = self.saved
        n_samples = X.shape[1]
        dLdx = 2 * (X - y) / n_samples
        return dLdx


class Adam:
    def __init__(
        self, params: List[Parameter], lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8
    ):
        self.params = list(params)
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.t = 1

    def step(self):
        for m, v, p in zip(self.m, self.v, self.params):
            if p.grad is None:
                continue

            m = self.beta1 * m + (1 - self.beta1) * p.grad
            v = self.beta2 * v + (1 - self.beta2) * p.grad**2

            m_corr = m / (1 - self.beta1**self.t)
            v_corr = v / (1 - self.beta2**self.t)

            p -= self.lr * m_corr / (np.sqrt(v_corr) + self.eps)
        self.t += 1

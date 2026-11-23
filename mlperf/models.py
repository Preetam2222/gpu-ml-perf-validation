"""Small, deterministic model zoo used as benchmark subjects.

Three architectures that stress different things: an MLP (large matmuls, memory-bound at small
batch), a convnet (conv + batchnorm), and a transformer encoder (attention + layernorm).
`scale="tiny"` gives CPU-test-sized versions; `scale="base"` gives sizes that are meaningful on a GPU.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
from torch import nn


@dataclass
class ModelSpec:
    name: str
    module: nn.Module
    input_shape: Tuple[int, ...]  # per sample, excluding batch


def build_mlp(scale: str) -> ModelSpec:
    d_in, hidden, d_out = {"tiny": (32, 64, 10), "base": (1024, 4096, 1000)}[scale]
    m = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                      nn.Linear(hidden, d_out))
    return ModelSpec("mlp", m, (d_in,))


def build_convnet(scale: str) -> ModelSpec:
    chans, size, classes = {"tiny": ((3, 8, 16), 16, 10), "base": ((3, 64, 128, 256), 64, 100)}[scale]
    layers = []
    for cin, cout in zip(chans[:-1], chans[1:]):
        layers += [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                   nn.Conv2d(cout, cout, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU()]
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(chans[-1], classes)]
    return ModelSpec("convnet", nn.Sequential(*layers), (3, size, size))


class _TransformerClassifier(nn.Module):
    def __init__(self, d: int, heads: int, layers: int, ff: int, classes: int):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x).mean(dim=1))


def build_transformer(scale: str) -> ModelSpec:
    d, heads, layers, ff, seq, classes = {
        "tiny": (32, 2, 1, 64, 8, 10),
        "base": (256, 4, 4, 1024, 64, 100),
    }[scale]
    return ModelSpec("transformer", _TransformerClassifier(d, heads, layers, ff, classes), (seq, d))


MODELS: Dict[str, Callable[[str], ModelSpec]] = {
    "mlp": build_mlp,
    "convnet": build_convnet,
    "transformer": build_transformer,
}


def build(name: str, scale: str = "base", seed: int = 1234) -> ModelSpec:
    """Deterministically initialised model in eval mode (fp32, CPU)."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    if scale not in ("tiny", "base"):
        raise ValueError("scale must be 'tiny' or 'base'")
    torch.manual_seed(seed)
    spec = MODELS[name](scale)
    spec.module.eval()
    return spec


def make_input(spec: ModelSpec, batch: int, seed: int = 4321) -> torch.Tensor:
    """Deterministic fp32 CPU input of shape (batch, *input_shape)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, *spec.input_shape, generator=g)


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

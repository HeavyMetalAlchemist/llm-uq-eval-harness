from __future__ import annotations

from typing import Type

from .base import BaseMetric

REGISTRY: dict[str, Type[BaseMetric]] = {}


def register_metric(cls: Type[BaseMetric]) -> Type[BaseMetric]:
    """Register a metric class. Can be used as a decorator or called directly."""
    REGISTRY[cls.name] = cls
    return cls


def get_metric(name: str) -> BaseMetric:
    if name not in REGISTRY:
        raise KeyError(f"Unknown metric '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]()


def load_builtins() -> None:
    """Ensure built-in metrics are registered. Called once at startup."""
    from . import builtin  # noqa: F401

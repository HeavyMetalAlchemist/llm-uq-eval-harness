from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    name: str
    scalars: dict[str, float] = field(default_factory=dict)
    # optional: extra data for report (e.g. arrays for plotting)
    extras: dict[str, Any] = field(default_factory=dict)


class BaseMetric(ABC):
    """Plugin interface for uncertainty metrics.

    To add a new metric:
    1. Subclass BaseMetric and set `name`.
    2. Implement `compute(df)` — receives the full results DataFrame,
       returns a MetricResult.
    3. Call `register_metric(MyMetric)` to make it available by name.
    """

    name: str

    @abstractmethod
    def compute(self, df) -> MetricResult:
        ...

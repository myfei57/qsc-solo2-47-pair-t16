"""Read model for the moisture system."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class WaterView:
    sensors: Mapping[str, float]
    samples: int
    average: float
    span: float
    edge_center: float
    generation: int
    alignment: Mapping[str, str]
    alignment_generation: int
    setpoints: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensors": dict(self.sensors),
            "samples": self.samples,
            "average": self.average,
            "span": self.span,
            "edge_center": self.edge_center,
            "generation": self.generation,
            "alignment": dict(self.alignment),
            "alignment_generation": self.alignment_generation,
            "setpoints": dict(self.setpoints),
        }

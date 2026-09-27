"""Read model for the cross direction profile control."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProfileView:
    actuators: int
    generation: int
    adjustments: int
    open_adjustments: int
    last_improved: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "actuators": self.actuators,
            "generation": self.generation,
            "adjustments": self.adjustments,
            "open_adjustments": self.open_adjustments,
            "last_improved": self.last_improved,
        }

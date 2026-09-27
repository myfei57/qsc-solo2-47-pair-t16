"""Cross direction moisture profile control: alignment, advice and review.

The water service only reads the profile; this service closes the loop. Actuators
are aligned one-to-one with the mapped sensors as a versioned parameter set, move
advice is computed against a target profile with a separate edge compensation, and
every adjustment is recorded with the alignment generation in force, so its effect
can be reviewed against fresh samples and traced back to the mapping that produced
it.
"""

from typing import Any, Mapping, Protocol, Sequence

from ..clock import IdSequence, LogicalClock, parse_id
from ..errors import AlignmentError, PreGateError, UnknownAdjustmentError
from ..event import EventBus, topics
from ..store import Journal, keys
from ..verdict import Band
from ..version import Confirmations, Generations
from . import calc
from .view import ProfileView

ALIGNMENT_NAME = "profile.alignment"
PROFILE_NAME = "profile"
TARGET_BAND = Band(2.0, 12.0)
DEFAULT_TARGET = 7.0


class MoistureSource(Protocol):
    """Anything that can report the sensor map and the moisture samples."""

    def map_sensors(self) -> dict[str, float]: ...

    def samples(self) -> tuple[dict[str, Any], ...]: ...


class ProfileService:
    """Actuator alignment, target profile advice and adjustment review."""

    def __init__(
        self,
        journal: Journal,
        bus: EventBus,
        clock: LogicalClock,
        generations: Generations,
        confirmations: Confirmations,
        water: MoistureSource,
    ) -> None:
        self._journal = journal
        self._bus = bus
        self._clock = clock
        self._generations = generations
        self._confirmations = confirmations
        self._water = water
        self._ids = IdSequence("adj")
        self._pairs: tuple[dict[str, Any], ...] = ()
        self._adjustments: list[dict[str, Any]] = []

    # ----------------------------------------------------------------- reading

    def alignment(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(pair) for pair in self._pairs)

    def generation(self) -> int:
        return self._generations.current(ALIGNMENT_NAME)

    def adjustments(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(record) for record in self._adjustments)

    def adjustment(self, adjustment_id: str) -> dict[str, Any]:
        return dict(self._find(adjustment_id))

    def suggest(self, target: float | None = None) -> dict[str, Any]:
        """Advised actuator moves against a target profile, edges compensated apart."""
        pairs = self._require_alignment()
        target = self._target(target)
        by_sensor = self._require_samples(pairs)
        return {
            "target": target,
            "generation": self.generation(),
            "edge_bias": calc.edge_bias(pairs, by_sensor),
            "moves": calc.moves(pairs, by_sensor, target),
            "tick": self._clock.tick,
        }

    def view(self) -> ProfileView:
        evaluated = [r for r in self._adjustments if r["status"] == "evaluated"]
        return ProfileView(
            actuators=len(self._pairs),
            generation=self.generation(),
            adjustments=len(self._adjustments),
            open_adjustments=len(self._adjustments) - len(evaluated),
            last_improved=evaluated[-1]["improved"] if evaluated else None,
        )

    # ------------------------------------------------------------------ writes

    def rebuild_alignment(
        self,
        pairs: Sequence[Mapping[str, str]],
        confirmation_id: str,
    ) -> dict[str, Any]:
        """Replace the actuator to sensor alignment under a confirmation."""
        checked = calc.validate_pairs(pairs, self._water.map_sensors())
        self._confirmations.redeem(confirmation_id, ALIGNMENT_NAME, PROFILE_NAME)
        generation = self._generations.advance(ALIGNMENT_NAME, "profile-alignment-generation")
        tick = self._clock.tick
        row = {"pairs": checked, "generation": generation, "tick": tick}
        with self._journal.transaction("profile-alignment") as batch:
            self._journal.append(keys.PROFILE_ALIGNMENT, row, batch, generation=generation)
        self._pairs = tuple(dict(pair) for pair in checked)
        self._bus.publish(topics.TOPIC_PROFILE_ALIGNMENT, generation)
        return dict(row)

    def adjust(
        self,
        target: float | None = None,
        reported: Sequence[Mapping[str, object]] | None = None,
    ) -> dict[str, Any]:
        """Record an adjustment: the moves made and the profile they started from."""
        pairs = self._require_alignment()
        target = self._target(target)
        by_sensor = self._require_samples(pairs)
        if reported is None:
            applied = calc.moves(pairs, by_sensor, target)
        else:
            applied = calc.check_moves(pairs, reported)
        record = {
            "adjustment_id": self._ids.next(),
            "generation": self.generation(),
            "target": target,
            "pairs": [dict(pair) for pair in pairs],
            "moves": applied,
            "before": calc.metrics(self._water.samples(), pairs, target),
            "status": "open",
            "tick": self._clock.tick,
        }
        self._adjustments = [*self._adjustments, record]
        self._persist("profile-adjust")
        self._bus.publish(topics.TOPIC_PROFILE_ADJUST, record["adjustment_id"])
        return dict(record)

    def evaluate(self, adjustment_id: str | None = None) -> dict[str, Any]:
        """Review an adjustment against fresh samples: better or not, and by how much."""
        record = self._find(adjustment_id)
        if record["status"] != "open":
            raise AlignmentError(
                f"adjustment {record['adjustment_id']} is already evaluated"
            )
        pairs = record["pairs"]
        self._require_samples(pairs)
        after = calc.metrics(self._water.samples(), pairs, float(record["target"]))
        updated = {
            **record,
            "after": after,
            "deltas": calc.deltas(record["before"], after),
            "improved": calc.improved(record["before"], after),
            "status": "evaluated",
            "evaluated_tick": self._clock.tick,
            "evaluated_generation": self.generation(),
        }
        self._adjustments = [
            updated if item["adjustment_id"] == record["adjustment_id"] else item
            for item in self._adjustments
        ]
        self._persist("profile-evaluate")
        self._bus.publish(topics.TOPIC_PROFILE_EVALUATED, updated["adjustment_id"])
        return dict(updated)

    def reload(self) -> ProfileView:
        """Restore the alignment and the adjustment log from the committed stream."""
        stored = self._journal.read(keys.PROFILE_ALIGNMENT)
        if stored is not None:
            self._pairs = tuple(dict(pair) for pair in stored["pairs"])
        rows = self._journal.read(keys.PROFILE_ADJUSTMENTS)
        self._adjustments = [dict(row) for row in rows] if rows else []
        for row in self._adjustments:
            number = parse_id(str(row["adjustment_id"]))
            if number is not None:
                self._ids.restore(number)
        return self.view()

    # ---------------------------------------------------------------- internals

    def _target(self, target: float | None) -> float:
        value = DEFAULT_TARGET if target is None else float(target)
        return TARGET_BAND.require(value, "profile target")

    def _require_alignment(self) -> tuple[dict[str, Any], ...]:
        if not self._pairs:
            raise PreGateError("profile: no actuator alignment is in force")
        return self._pairs

    def _require_samples(self, pairs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        by_sensor = calc.latest_by_sensor(self._water.samples())
        for pair in pairs:
            sensor = str(pair["sensor"])
            if sensor not in by_sensor:
                raise PreGateError(f"profile: no sample for sensor {sensor}")
        return by_sensor

    def _find(self, adjustment_id: str | None) -> dict[str, Any]:
        if adjustment_id is None:
            for record in reversed(self._adjustments):
                if record["status"] == "open":
                    return record
            raise UnknownAdjustmentError("no open profile adjustment")
        for record in self._adjustments:
            if record["adjustment_id"] == adjustment_id:
                return record
        raise UnknownAdjustmentError(f"unknown adjustment {adjustment_id}")

    def _persist(self, prefix: str) -> None:
        with self._journal.transaction(prefix) as batch:
            self._journal.append(keys.PROFILE_ADJUSTMENTS, self._adjustments, batch)

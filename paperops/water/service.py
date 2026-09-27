"""Moisture sensing, the sensor map and the cross direction profile.

The sensor map is a parameter set with a generation. Sampling by sensor name resolves
the position through the map, and rebuilding the map drops the cached profile, so a
profile computed against an old map can never be served after a scanner change.

The actuator side closes the loop: an alignment pairs each actuator with exactly one
mapped sensor and carries its own generation plus the sensor map generation it was
derived from. Suggestions are computed against a target profile with a separate edge
compensation, every applied move freezes the profile it was based on, and an
evaluation compares the before and after shapes so an adjustment round always shows
whether the profile improved and which alignment generation it steered with.
"""

from typing import Any, Mapping, Sequence

from ..clock import LogicalClock
from ..errors import (
    OutOfRangeError,
    PreGateError,
    StaleArtifactError,
    UnknownActuatorError,
    UnknownSensorError,
)
from ..event import Event, EventBus, topics
from ..store import Journal, keys
from ..verdict import Band, VerdictHistory, Window, WindowEvaluator
from ..verdict import ABOVE_MODE
from ..version import Confirmations, Generations
from . import calc
from .view import WaterView

MAP_NAME = "water.map"
WATER_NAME = "water"
ALIGNMENT_NAME = "water.alignment"
SAMPLE_LIMIT = 60
EVALUATION_LIMIT = 30
MOISTURE_BAND = Band(2.0, 12.0)
SETPOINT_BAND = Band(0.0, 100.0)
WINDOW_SIZE = 3
RESPONSE_GAIN = 4.0
EDGE_GAIN = 1.0
EDGE_MARGIN = 0.5


class WaterService:
    """Sensor map, moisture samples and the cross direction profile."""

    def __init__(
        self,
        journal: Journal,
        bus: EventBus,
        clock: LogicalClock,
        generations: Generations,
        confirmations: Confirmations,
    ) -> None:
        self._journal = journal
        self._bus = bus
        self._clock = clock
        self._generations = generations
        self._confirmations = confirmations
        self._band = MOISTURE_BAND
        self._setpoint_band = SETPOINT_BAND
        self._history = VerdictHistory("water.moisture")
        self._adjustment_history = VerdictHistory("water.adjustment")
        self._sustained = WindowEvaluator(Window(WINDOW_SIZE), 12.0, ABOVE_MODE, WINDOW_SIZE)
        self._sensors = calc.default_sensors()
        self._alignment: dict[str, str] = {}
        self._setpoints: dict[str, float] = {}
        self._profile_cache: list[dict[str, float]] | None = None

    # ----------------------------------------------------------------- reading

    def map_sensors(self) -> dict[str, float]:
        return dict(self._sensors)

    def generation(self) -> int:
        row = self._journal.read(keys.WATER_MAP)
        if row is None:
            return 0
        return int(row["generation"])

    def samples(self) -> tuple[dict[str, Any], ...]:
        rows = self._journal.read(keys.WATER_SAMPLES)
        if rows is None:
            return ()
        return tuple(rows)

    def values(self) -> tuple[float, ...]:
        return tuple(float(row["moisture"]) for row in self.samples())

    def average(self) -> float:
        return calc.average(self.values())

    def span(self) -> float:
        return calc.span(self.values())

    def edge_center(self) -> float:
        edges = [
            float(row["moisture"])
            for row in self.samples()
            if row["name"] in ("front", "back")
        ]
        center = next(
            (float(row["moisture"]) for row in self.samples() if row["name"] == "middle"),
            0.0,
        )
        if not edges:
            edges = list(self.values())
        return calc.edge_center(edges, center)

    def history(self) -> VerdictHistory:
        return self._history

    def adjustment_history(self) -> VerdictHistory:
        return self._adjustment_history

    def alignment(self) -> dict[str, str]:
        return dict(self._alignment)

    def alignment_generation(self) -> int:
        row = self._journal.read(keys.WATER_ALIGNMENT)
        if row is None:
            return 0
        return int(row["generation"])

    def alignment_positions(self) -> dict[str, float]:
        """Cross direction position of each aligned actuator, through the sensor map."""
        return {
            actuator: self._sensors[sensor]
            for actuator, sensor in self._alignment.items()
            if sensor in self._sensors
        }

    def setpoints(self) -> dict[str, float]:
        return dict(self._setpoints)

    def suggestion(self) -> dict[str, Any] | None:
        row = self._journal.read(keys.WATER_SUGGESTION)
        return None if row is None else dict(row)

    def adjustment(self) -> dict[str, Any] | None:
        row = self._journal.read(keys.WATER_ADJUSTMENT)
        return None if row is None else dict(row)

    def evaluations(self) -> tuple[dict[str, Any], ...]:
        rows = self._journal.read(keys.WATER_EVALUATIONS)
        if rows is None:
            return ()
        return tuple(dict(row) for row in rows)

    def profile(self) -> list[dict[str, float]]:
        """Cross direction profile, computed once and reused until the map moves."""
        if self._profile_cache is not None:
            return [dict(point) for point in self._profile_cache]
        points = sorted(
            (float(row["position"]), float(row["moisture"]))
            for row in self.samples()
        )
        self._profile_cache = [
            {"position": position, "moisture": moisture} for position, moisture in points
        ]
        return [dict(point) for point in self._profile_cache]

    def at(self, position: float) -> float:
        """Interpolated moisture at a cross direction position."""
        points = [
            (float(row["position"]), float(row["moisture"])) for row in self.samples()
        ]
        return calc.interpolate(points, position)

    def moisture(self) -> float:
        return self.average()

    def moisture_verdict(self, reading: float) -> str:
        result = calc.verdict(reading, self._band.low, self._band.high)
        self._history.record(
            result,
            self._journal.watermark,
            self._clock.tick,
            f"reading={reading}",
        )
        return result

    def sustained_dry(self, reading: float):
        """Verdict that only holds once a whole window of readings is above target."""
        return self._sustained.push(reading)

    def view(self) -> WaterView:
        return WaterView(
            sensors=self.map_sensors(),
            samples=len(self.samples()),
            average=self.average(),
            span=self.span(),
            edge_center=self.edge_center(),
            generation=self.generation(),
            alignment=self.alignment(),
            alignment_generation=self.alignment_generation(),
            setpoints=self.setpoints(),
        )

    # ------------------------------------------------------------------ writes

    def rebuild_map(self, sensors: Mapping[str, float], confirmation_id: str) -> dict[str, Any]:
        """Replace the sensor map under a confirmation and a new generation."""
        checked = calc.validate_sensors(sensors)
        self._confirmations.redeem(confirmation_id, MAP_NAME, WATER_NAME)
        generation = self._generations.advance(MAP_NAME, "water-map-generation")
        tick = self._clock.tick
        row = {"sensors": checked, "generation": generation, "tick": tick}
        with self._journal.transaction("water-map") as batch:
            self._journal.append(keys.WATER_MAP, row, batch, generation=generation)
        self._sensors = dict(checked)
        self._profile_cache = None
        self._bus.publish(topics.TOPIC_WATER_MAP, generation)
        return dict(row)

    def replace_scanner(self) -> int:
        """Move the generation on so a profile built on the old map is refused."""
        generation = self._generations.advance(MAP_NAME, "water-scanner")
        self._profile_cache = None
        self._bus.publish(topics.TOPIC_WATER_MAP, generation)
        return generation

    def rebuild_alignment(
        self, pairs: Mapping[str, str], confirmation_id: str
    ) -> dict[str, Any]:
        """Pair actuators with sensors one to one, under a confirmation."""
        checked = calc.validate_alignment(pairs, self._sensors)
        self._confirmations.redeem(confirmation_id, ALIGNMENT_NAME, WATER_NAME)
        generation = self._generations.advance(ALIGNMENT_NAME, "water-alignment-generation")
        row = {
            "pairs": checked,
            "map_generation": self.generation(),
            "generation": generation,
            "tick": self._clock.tick,
        }
        with self._journal.transaction("water-alignment") as batch:
            self._journal.append(keys.WATER_ALIGNMENT, row, batch, generation=generation)
        self._alignment = dict(checked)
        self._bus.publish(topics.TOPIC_WATER_ALIGNMENT, generation)
        return dict(row)

    def suggest(
        self,
        target: float | None = None,
        targets: Sequence[Mapping[str, float]] | None = None,
        gain: float = RESPONSE_GAIN,
        edge_gain: float = EDGE_GAIN,
        edge_margin: float = EDGE_MARGIN,
    ) -> dict[str, Any]:
        """Advise one move per actuator against a target profile.

        Edge actuators carry a separate compensation for the systematic
        edge-to-center bias, on top of the error-proportional move.
        """
        alignment = self._require_alignment()
        points = self._profile_points()
        if not points:
            raise PreGateError("no moisture samples to steer a suggestion from")
        gain = self.negative(gain, "response gain")
        edge_gain = self.negative(edge_gain, "edge gain")
        edge_margin = self.negative(edge_margin, "edge margin")
        target_points = self._target_points(target, targets)
        before = calc.profile_stats(points, target_points, edge_margin)
        edge_excess = max(0.0, before["edge_center"])
        moves = calc.suggest_moves(
            points,
            self.alignment_positions(),
            self._setpoints,
            target_points,
            gain,
            edge_gain,
            edge_margin,
            edge_excess,
        )
        row = {
            "moves": moves,
            "targets": [[position, moisture] for position, moisture in target_points],
            "gain": gain,
            "edge_gain": edge_gain,
            "edge_margin": edge_margin,
            "edge_excess": edge_excess,
            "before": before,
            "generation": int(alignment["generation"]),
            "tick": self._clock.tick,
        }
        with self._journal.transaction("water-suggestion") as batch:
            self._journal.append(
                keys.WATER_SUGGESTION, row, batch, generation=int(alignment["generation"])
            )
        self._bus.publish(topics.TOPIC_WATER_SUGGESTION, len(moves))
        return dict(row)

    def apply_suggestion(self) -> dict[str, Any]:
        """Apply the latest suggestion; a suggestion from an old alignment is refused."""
        suggestion = self.suggestion()
        if suggestion is None:
            raise PreGateError("no suggestion to apply")
        self._require_alignment()
        self._generations.require_current(
            ALIGNMENT_NAME, int(suggestion["generation"]), "water suggestion"
        )
        setpoints = {
            str(move["actuator"]): float(move["suggested"]) for move in suggestion["moves"]
        }
        return self._record_adjustment(
            setpoints,
            suggestion["targets"],
            "suggestion",
            suggestion["before"],
            float(suggestion["edge_margin"]),
        )

    def set_actuators(
        self,
        setpoints: Mapping[str, float],
        target: float | None = None,
        targets: Sequence[Mapping[str, float]] | None = None,
    ) -> dict[str, Any]:
        """Record a manual actuator move, freezing the profile it was based on."""
        self._require_alignment()
        if not setpoints:
            raise ValueError("at least one actuator setpoint is required")
        checked: dict[str, float] = {}
        for name, value in setpoints.items():
            name = str(name)
            if name not in self._alignment:
                raise UnknownActuatorError(f"actuator {name} is not aligned")
            checked[name] = self._setpoint_band.require(
                float(value), f"actuator {name} setpoint"
            )
        points = self._profile_points()
        if not points:
            raise PreGateError("no moisture samples to base an adjustment on")
        target_points = self._target_points(target, targets)
        before = calc.profile_stats(points, target_points, EDGE_MARGIN)
        return self._record_adjustment(checked, target_points, "manual", before, EDGE_MARGIN)

    def evaluate(self) -> dict[str, Any]:
        """Judge the latest adjustment against the profile it froze."""
        adjustment = self.adjustment()
        if adjustment is None:
            raise PreGateError("no actuator adjustment to evaluate")
        points = self._profile_points()
        if not points:
            raise PreGateError("no moisture samples to evaluate against")
        target_points = [
            (float(position), float(moisture)) for position, moisture in adjustment["targets"]
        ]
        after = calc.profile_stats(
            points, target_points, float(adjustment["edge_margin"])
        )
        verdict = calc.evaluate(adjustment["before"], after)
        row = {
            "adjustment_tick": int(adjustment["tick"]),
            "generation": int(adjustment["generation"]),
            "source": adjustment["source"],
            "before": adjustment["before"],
            "after": after,
            **verdict,
            "tick": self._clock.tick,
        }
        rows = list(self.evaluations())
        rows.append(row)
        rows = rows[-EVALUATION_LIMIT:]
        with self._journal.transaction("water-evaluation") as batch:
            self._journal.append(
                keys.WATER_EVALUATIONS, rows, batch, generation=int(adjustment["generation"])
            )
        self._adjustment_history.record(
            str(verdict["verdict"]),
            self._journal.watermark,
            self._clock.tick,
            f"mean_abs_error {adjustment['before']['mean_abs_error']:.2f} -> "
            f"{after['mean_abs_error']:.2f}",
        )
        self._bus.publish(topics.TOPIC_WATER_EVALUATION, verdict["verdict"])
        return dict(row)

    def add_sample(self, name: str, moisture: float) -> dict[str, Any]:
        """Sample a named sensor; the name has to exist in the current map."""
        if name not in self._sensors:
            raise UnknownSensorError(f"sensor {name} is not mapped")
        return self.add_reading(name, self._sensors[name], moisture)

    def add_reading(self, name: str, position: float, moisture: float) -> dict[str, Any]:
        """Record a moisture reading at a known cross direction position."""
        self._band.require(moisture, "moisture reading")
        rows = list(self.samples())
        rows.append(
            {
                "name": name,
                "position": float(position),
                "moisture": float(moisture),
                "tick": self._clock.tick,
            }
        )
        rows = rows[-SAMPLE_LIMIT:]
        with self._journal.transaction("water-sample") as batch:
            self._journal.append(keys.WATER_SAMPLES, rows, batch)
        self._profile_cache = None
        self._bus.publish(topics.TOPIC_WATER_SAMPLE, name)
        return {"name": name, "position": float(position), "moisture": float(moisture)}

    def record_profile(self) -> dict[str, Any]:
        """Freeze the current profile as a recorded curve."""
        points = self.profile()
        row = {
            "points": points,
            "average": self.average(),
            "span": self.span(),
            "tick": self._clock.tick,
        }
        with self._journal.transaction("water-profile") as batch:
            self._journal.append(keys.WATER_PROFILE, row, batch)
        self._bus.publish(topics.TOPIC_WATER_PROFILE, len(points))
        return dict(row)

    def reload(self) -> WaterView:
        stored = self._journal.read(keys.WATER_MAP)
        if stored is not None:
            self._sensors = {str(k): float(v) for k, v in stored["sensors"].items()}
        alignment = self._journal.read(keys.WATER_ALIGNMENT)
        if alignment is not None:
            self._alignment = {str(k): str(v) for k, v in alignment["pairs"].items()}
        actuators = self._journal.read(keys.WATER_ACTUATORS)
        if actuators is not None:
            self._setpoints = {str(k): float(v) for k, v in actuators["setpoints"].items()}
        self._profile_cache = None
        return self.view()

    def wire(self) -> None:
        """Drop the cached profile whenever the sensor map moves."""
        self._bus.subscribe(topics.TOPIC_WATER_MAP, self._on_map_change)

    # -------------------------------------------------------------- arithmetic

    def average_of(self, readings: Sequence[float]) -> float:
        return calc.average(readings)

    def span_of(self, readings: Sequence[float]) -> float:
        return calc.span(readings)

    def deviation(self, reading: float) -> float:
        return float(reading) - self.average()

    def verdict_of(self, reading: float) -> str:
        return calc.verdict(reading, self._band.low, self._band.high)

    def verify_band(self, value: float, label: str) -> float:
        return self._band.require(value, label)

    def negative(self, value: float, label: str) -> float:
        if value < 0:
            raise OutOfRangeError(f"{label} must not be negative")
        return float(value)

    # ---------------------------------------------------------------- internals

    def _require_alignment(self) -> dict[str, Any]:
        """The committed alignment, refused once the sensor map has moved on."""
        row = self._journal.read(keys.WATER_ALIGNMENT)
        if row is None:
            raise UnknownActuatorError("no actuator alignment is in force")
        map_generation = self.generation()
        if int(row["map_generation"]) != map_generation:
            raise StaleArtifactError(
                f"water alignment was built against sensor map generation "
                f"{row['map_generation']} but water.map is at {map_generation}"
            )
        return row

    def _profile_points(self) -> list[tuple[float, float]]:
        """The current scan: the latest reading per sensor, ordered by position."""
        latest: dict[str, tuple[float, float]] = {}
        for row in self.samples():
            latest[str(row["name"])] = (float(row["position"]), float(row["moisture"]))
        return sorted(latest.values())

    def _target_points(
        self,
        target: float | None,
        targets: Sequence[Mapping[str, float]] | None,
    ) -> list[tuple[float, float]]:
        if target is None and not targets:
            target = self.average()
        points = calc.normalize_targets(target, targets)
        for _, moisture in points:
            self._band.require(moisture, "target moisture")
        return points

    def _record_adjustment(
        self,
        setpoints: Mapping[str, float],
        target_points: Sequence[Sequence[float]],
        source: str,
        before: Mapping[str, float],
        edge_margin: float,
    ) -> dict[str, Any]:
        merged = {**self._setpoints, **{str(k): float(v) for k, v in setpoints.items()}}
        generation = self.alignment_generation()
        tick = self._clock.tick
        actuator_row = {
            "setpoints": merged,
            "generation": generation,
            "tick": tick,
            "source": source,
        }
        adjustment = {
            "setpoints": merged,
            "targets": [[float(p), float(m)] for p, m in target_points],
            "before": dict(before),
            "edge_margin": float(edge_margin),
            "generation": generation,
            "tick": tick,
            "source": source,
        }
        with self._journal.transaction("water-actuators") as batch:
            self._journal.append(keys.WATER_ACTUATORS, actuator_row, batch, generation=generation)
            self._journal.append(keys.WATER_ADJUSTMENT, adjustment, batch, generation=generation)
        self._setpoints = dict(merged)
        self._bus.publish(topics.TOPIC_WATER_ACTUATORS, len(merged))
        return dict(adjustment)

    def _on_map_change(self, event: Event) -> None:
        self._profile_cache = None

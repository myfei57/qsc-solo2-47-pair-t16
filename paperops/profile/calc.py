"""Cross direction profile control arithmetic.

An alignment pairs each actuator with exactly one sensor. Move advice works from
the signed error against the target profile; a positive move asks for more drying
at that position. Edge zones carry a separate compensation term so a systematic
edge-to-centre bias is fought by the edge actuators alone.
"""

import math
from typing import Mapping, Sequence

from ..errors import AlignmentError, UnknownActuatorError, UnknownSensorError

CENTER = "center"
EDGE = "edge"
ZONES = (CENTER, EDGE)

DEFAULT_GAIN = 10.0
DEFAULT_EDGE_GAIN = 5.0
MAX_MOVE = 100.0


def validate_pairs(
    pairs: Sequence[Mapping[str, str]],
    sensors: Mapping[str, float],
) -> list[dict[str, object]]:
    """Check that actuators and sensors pair up one-to-one against the map."""
    if not pairs:
        raise AlignmentError("an alignment must declare at least one pair")
    checked: list[dict[str, object]] = []
    actuators: set[str] = set()
    mapped: set[str] = set()
    for pair in pairs:
        actuator = str(pair["actuator"])
        sensor = str(pair["sensor"])
        zone = str(pair.get("zone", CENTER))
        if actuator in actuators:
            raise AlignmentError(f"actuator {actuator} is aligned twice")
        if sensor in mapped:
            raise AlignmentError(f"sensor {sensor} is aligned twice")
        if sensor not in sensors:
            raise UnknownSensorError(f"sensor {sensor} is not mapped")
        if zone not in ZONES:
            raise AlignmentError(f"zone {zone} is not one of {ZONES}")
        actuators.add(actuator)
        mapped.add(sensor)
        checked.append(
            {
                "actuator": actuator,
                "sensor": sensor,
                "zone": zone,
                "position": float(sensors[sensor]),
            }
        )
    return checked


def latest_by_sensor(samples: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Most recent moisture reading per sensor name."""
    latest: dict[str, float] = {}
    for row in samples:
        latest[str(row["name"])] = float(row["moisture"])
    return latest


def edge_bias(pairs: Sequence[Mapping[str, object]], by_sensor: Mapping[str, float]) -> float:
    """Mean moisture of the edge zones minus that of the centre zones."""
    edge = [
        by_sensor[str(pair["sensor"])]
        for pair in pairs
        if pair["zone"] == EDGE and str(pair["sensor"]) in by_sensor
    ]
    center = [
        by_sensor[str(pair["sensor"])]
        for pair in pairs
        if pair["zone"] == CENTER and str(pair["sensor"]) in by_sensor
    ]
    if not edge or not center:
        return 0.0
    return sum(edge) / len(edge) - sum(center) / len(center)


def clamp(move: float, limit: float = MAX_MOVE) -> float:
    return max(-float(limit), min(float(limit), float(move)))


def moves(
    pairs: Sequence[Mapping[str, object]],
    by_sensor: Mapping[str, float],
    target: float,
    gain: float = DEFAULT_GAIN,
    edge_gain: float = DEFAULT_EDGE_GAIN,
) -> list[dict[str, object]]:
    """Suggested actuator moves against a target, edges compensated separately."""
    bias = edge_bias(pairs, by_sensor)
    advised: list[dict[str, object]] = []
    for pair in pairs:
        sensor = str(pair["sensor"])
        measured = by_sensor[sensor]
        error = measured - float(target)
        move = float(gain) * error
        if pair["zone"] == EDGE:
            move += float(edge_gain) * bias
        advised.append(
            {
                "actuator": str(pair["actuator"]),
                "sensor": sensor,
                "zone": str(pair["zone"]),
                "position": float(pair["position"]),
                "measured": measured,
                "target": float(target),
                "error": error,
                "move": clamp(move),
            }
        )
    return advised


def check_moves(
    pairs: Sequence[Mapping[str, object]],
    reported: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Shape operator-reported moves onto the alignment; unknown actuators refuse."""
    aligned = {str(pair["actuator"]): pair for pair in pairs}
    applied: dict[str, float] = {}
    for move in reported:
        actuator = str(move["actuator"])
        if actuator not in aligned:
            raise UnknownActuatorError(f"actuator {actuator} is not aligned")
        applied[actuator] = clamp(float(move["move"]))
    checked: list[dict[str, object]] = []
    for actuator, pair in aligned.items():
        checked.append(
            {
                "actuator": actuator,
                "sensor": str(pair["sensor"]),
                "zone": str(pair["zone"]),
                "position": float(pair["position"]),
                "move": applied.get(actuator, 0.0),
            }
        )
    return checked


def average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def span(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)


def rms(values: Sequence[float], target: float) -> float:
    """Root mean square error of a profile against its target."""
    if not values:
        return 0.0
    return math.sqrt(sum((float(value) - float(target)) ** 2 for value in values) / len(values))


def metrics(
    samples: Sequence[Mapping[str, object]],
    pairs: Sequence[Mapping[str, object]],
    target: float,
) -> dict[str, float]:
    """Health of the current profile against its target, for before/after review."""
    by_sensor = latest_by_sensor(samples)
    values = list(by_sensor.values())
    edge = [
        by_sensor[str(pair["sensor"])]
        for pair in pairs
        if pair["zone"] == EDGE and str(pair["sensor"]) in by_sensor
    ]
    center = [
        by_sensor[str(pair["sensor"])]
        for pair in pairs
        if pair["zone"] == CENTER and str(pair["sensor"]) in by_sensor
    ]
    edge_center = 0.0
    if edge and center:
        edge_center = sum(edge) / len(edge) - sum(center) / len(center)
    return {
        "average": average(values),
        "span": span(values),
        "edge_center": edge_center,
        "rms": rms(values, target),
    }


def deltas(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    return {key: float(after[key]) - float(before[key]) for key in before}


def improved(before: Mapping[str, float], after: Mapping[str, float]) -> bool:
    """True when the profile sits closer to its target than before the moves."""
    return float(after["rms"]) < float(before["rms"])

"""Moisture profile arithmetic."""

from typing import Mapping, Sequence

from ..errors import UnknownSensorError

LOW = "low"
HIGH = "high"
IN_BAND = "in_band"

IMPROVED = "improved"
WORSENED = "worsened"
UNCHANGED = "unchanged"

SETPOINT_LOW = 0.0
SETPOINT_HIGH = 100.0
EVAL_EPSILON = 0.01


def default_sensors() -> dict[str, float]:
    """Positions of a freshly commissioned scanner, in metres from the front edge."""
    return {"front": 0.4, "middle": 3.1, "back": 5.8}


def validate_sensors(sensors: Mapping[str, float]) -> dict[str, float]:
    """Reject an empty map and keep positions ordered and non-negative."""
    if not sensors:
        raise ValueError("a sensor map must declare at least one sensor")
    checked: dict[str, float] = {}
    for name, position in sensors.items():
        value = float(position)
        if value < 0:
            raise ValueError(f"sensor {name} has a negative position")
        checked[str(name)] = value
    return checked


def average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def span(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)


def edge_center(edges: Sequence[float], center: float) -> float:
    """Difference between the mean of the edges and the centre of the web."""
    return average(edges) - float(center)


def verdict(reading: float, low: float, high: float) -> str:
    if reading < low:
        return LOW
    if reading > high:
        return HIGH
    return IN_BAND


def interpolate(points: Sequence[tuple[float, float]], position: float) -> float:
    """Linear interpolation of a profile at a position, clamped to the ends."""
    if not points:
        return 0.0
    ordered = sorted(points)
    if position <= ordered[0][0]:
        return ordered[0][1]
    if position >= ordered[-1][0]:
        return ordered[-1][1]
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        if x0 <= position <= x1:
            if x1 == x0:
                return y0
            ratio = (position - x0) / (x1 - x0)
            return y0 + (y1 - y0) * ratio
    return ordered[-1][1]


def clamp(value: float, low: float = SETPOINT_LOW, high: float = SETPOINT_HIGH) -> float:
    return min(max(float(value), low), high)


def validate_alignment(pairs: Mapping[str, str], sensors: Mapping[str, float]) -> dict[str, str]:
    """Pair each actuator with exactly one mapped sensor, and no sensor twice."""
    if not pairs:
        raise ValueError("an alignment must declare at least one actuator")
    checked: dict[str, str] = {}
    used: set[str] = set()
    for actuator, sensor in pairs.items():
        actuator = str(actuator)
        sensor = str(sensor)
        if not actuator:
            raise ValueError("an actuator name must not be blank")
        if sensor not in sensors:
            raise UnknownSensorError(f"sensor {sensor} is not mapped")
        if sensor in used:
            raise ValueError(f"sensor {sensor} is aligned to more than one actuator")
        used.add(sensor)
        checked[actuator] = sensor
    return checked


def normalize_targets(
    target: float | None,
    targets: Sequence[Mapping[str, float]] | None,
) -> list[tuple[float, float]]:
    """A target curve as interpolation points; a flat target is a single point."""
    if targets:
        points = sorted(
            (float(point["position"]), float(point["moisture"])) for point in targets
        )
        if not points:
            raise ValueError("a target profile must declare at least one point")
        return points
    if target is None:
        raise ValueError("a target moisture is required")
    return [(0.0, float(target))]


def edge_positions(
    points: Sequence[tuple[float, float]], margin: float
) -> tuple[float, float]:
    """Positions at or inside the edge margin of a profile, as a (low, high) band."""
    ordered = sorted(points)
    return ordered[0][0] + margin, ordered[-1][0] - margin


def is_edge(position: float, points: Sequence[tuple[float, float]], margin: float) -> bool:
    if not points:
        return False
    low, high = edge_positions(points, margin)
    return position <= low or position >= high


def profile_stats(
    points: Sequence[tuple[float, float]],
    target_points: Sequence[tuple[float, float]],
    edge_margin: float,
) -> dict[str, float]:
    """Shape of a measured profile against a target curve."""
    ordered = sorted((float(position), float(moisture)) for position, moisture in points)
    if not ordered:
        return {
            "samples": 0,
            "average": 0.0,
            "span": 0.0,
            "edge_center": 0.0,
            "mean_abs_error": 0.0,
        }
    values = [moisture for _, moisture in ordered]
    low, high = edge_positions(ordered, edge_margin)
    edges = [moisture for position, moisture in ordered if position <= low or position >= high]
    middle = (ordered[0][0] + ordered[-1][0]) / 2.0
    center = interpolate(ordered, middle)
    errors = [
        abs(moisture - interpolate(target_points, position))
        for position, moisture in ordered
    ]
    return {
        "samples": len(ordered),
        "average": average(values),
        "span": span(values),
        "edge_center": edge_center(edges, center),
        "mean_abs_error": average(errors),
    }


def suggest_moves(
    profile: Sequence[tuple[float, float]],
    actuators: Mapping[str, float],
    setpoints: Mapping[str, float],
    target_points: Sequence[tuple[float, float]],
    gain: float,
    edge_gain: float,
    edge_margin: float,
    edge_excess: float,
) -> list[dict[str, float]]:
    """One advised move per actuator; edge zones carry a separate compensation."""
    ordered = sorted(profile)
    moves: list[dict[str, float]] = []
    for name, position in sorted(actuators.items(), key=lambda item: item[1]):
        measured = interpolate(ordered, position)
        want = interpolate(target_points, position)
        error = measured - want
        edge = is_edge(position, ordered, edge_margin)
        boost = float(edge_gain) * edge_excess if edge else 0.0
        move = float(gain) * error + boost
        current = float(setpoints.get(name, 0.0))
        moves.append(
            {
                "actuator": name,
                "position": float(position),
                "measured": measured,
                "target": want,
                "error": error,
                "edge": edge,
                "edge_boost": boost,
                "move": move,
                "current": current,
                "suggested": clamp(current + move),
            }
        )
    return moves


def evaluate(
    before: Mapping[str, float], after: Mapping[str, float]
) -> dict[str, float | str]:
    """Judge an adjustment by the change in mean absolute error against target."""
    delta_error = float(after["mean_abs_error"]) - float(before["mean_abs_error"])
    if delta_error < -EVAL_EPSILON:
        verdict = IMPROVED
    elif delta_error > EVAL_EPSILON:
        verdict = WORSENED
    else:
        verdict = UNCHANGED
    return {
        "verdict": verdict,
        "mean_abs_error_delta": delta_error,
        "span_delta": float(after["span"]) - float(before["span"]),
        "edge_center_delta": float(after["edge_center"]) - float(before["edge_center"]),
    }

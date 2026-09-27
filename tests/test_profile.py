"""Actuator alignment, target profile advice and adjustment review."""

import pytest

from paperops.errors import AlignmentError, PreGateError, UnknownActuatorError

from support import build, issue_confirmation, ok, refused

PAIRS = [
    {"actuator": "act-front", "sensor": "front", "zone": "edge"},
    {"actuator": "act-mid", "sensor": "middle"},
    {"actuator": "act-back", "sensor": "back", "zone": "edge"},
]


def align(console, pairs=None):
    return ok(
        console,
        "POST",
        "/api/profile/alignment",
        {
            "pairs": PAIRS if pairs is None else pairs,
            "confirmation_id": issue_confirmation(console, "profile.alignment", "profile"),
        },
    )


def scan(console, front, middle, back):
    ok(console, "POST", "/api/water/sample", {"name": "front", "moisture": front})
    ok(console, "POST", "/api/water/sample", {"name": "middle", "moisture": middle})
    ok(console, "POST", "/api/water/sample", {"name": "back", "moisture": back})


def test_alignment_binds_a_confirmation_and_a_generation(console, services):
    body = align(console)
    assert body["generation"] == 1
    assert services.profile.generation() == 1
    pairs = ok(console, "GET", "/api/profile/alignment")
    assert pairs["generation"] == 1
    assert [pair["actuator"] for pair in pairs["pairs"]] == ["act-front", "act-mid", "act-back"]
    assert pairs["pairs"][0]["zone"] == "edge"
    assert pairs["pairs"][1]["zone"] == "center"


def test_alignment_must_be_one_to_one(console, services):
    duplicated = [PAIRS[0], dict(PAIRS[0])]
    assert refused(
        console,
        "POST",
        "/api/profile/alignment",
        {
            "pairs": duplicated,
            "confirmation_id": issue_confirmation(console, "profile.alignment", "profile"),
        },
    ).status == 422
    shared_sensor = [PAIRS[0], {"actuator": "act-other", "sensor": "front"}]
    assert refused(
        console,
        "POST",
        "/api/profile/alignment",
        {
            "pairs": shared_sensor,
            "confirmation_id": issue_confirmation(console, "profile.alignment", "profile"),
        },
    ).status == 422
    unknown = [{"actuator": "act-x", "sensor": "edge"}]
    assert refused(
        console,
        "POST",
        "/api/profile/alignment",
        {
            "pairs": unknown,
            "confirmation_id": issue_confirmation(console, "profile.alignment", "profile"),
        },
    ).status == 404
    with pytest.raises(AlignmentError):
        services.profile.rebuild_alignment([], "any")


def test_advice_and_adjustment_need_an_alignment(console, services):
    scan(console, 8.0, 6.0, 8.0)
    assert refused(console, "GET", "/api/profile/suggest?target=6").status == 409
    assert refused(console, "POST", "/api/profile/adjust", {"target": 6}).status == 409
    with pytest.raises(PreGateError):
        services.profile.suggest(6.0)


def test_advice_needs_a_full_scan(console, services):
    align(console)
    ok(console, "POST", "/api/water/sample", {"name": "front", "moisture": 8.0})
    assert refused(console, "GET", "/api/profile/suggest?target=6").status == 409


def test_suggestion_compensates_the_edges_separately(console, services):
    align(console)
    scan(console, 8.0, 6.0, 8.0)
    body = ok(console, "GET", "/api/profile/suggest?target=6")
    assert body["generation"] == 1
    assert body["edge_bias"] == 2.0
    moves = {move["actuator"]: move["move"] for move in body["moves"]}
    assert moves["act-mid"] == 0.0
    assert moves["act-front"] == 30.0
    assert moves["act-back"] == 30.0
    for move in body["moves"]:
        assert move["target"] == 6.0
        assert move["measured"] == {"act-front": 8.0, "act-mid": 6.0, "act-back": 8.0}[move["actuator"]]


def test_adjustment_review_shows_improvement_and_mapping_version(console, services):
    align(console)
    scan(console, 8.0, 6.0, 8.0)
    record = ok(console, "POST", "/api/profile/adjust", {"target": 6})
    assert record["status"] == "open"
    assert record["generation"] == 1
    assert record["adjustment_id"] == "adj-000001"
    assert record["before"]["edge_center"] == 2.0
    assert record["before"]["rms"] > 0
    scan(console, 6.5, 6.0, 6.5)
    reviewed = ok(console, "POST", "/api/profile/evaluate", {})
    assert reviewed["status"] == "evaluated"
    assert reviewed["improved"] is True
    assert reviewed["after"]["rms"] < reviewed["before"]["rms"]
    assert reviewed["deltas"]["edge_center"] == -1.5
    assert reviewed["generation"] == 1
    assert reviewed["evaluated_generation"] == 1


def test_adjustment_review_can_show_no_improvement(console, services):
    align(console)
    scan(console, 8.0, 6.0, 8.0)
    ok(console, "POST", "/api/profile/adjust", {"target": 6})
    scan(console, 9.0, 6.0, 9.0)
    reviewed = ok(console, "POST", "/api/profile/evaluate", {})
    assert reviewed["improved"] is False
    assert reviewed["after"]["rms"] > reviewed["before"]["rms"]


def test_adjustment_is_evaluated_only_once(console, services):
    align(console)
    scan(console, 8.0, 6.0, 8.0)
    record = ok(console, "POST", "/api/profile/adjust", {"target": 6})
    ok(console, "POST", "/api/profile/evaluate", {})
    assert refused(console, "POST", "/api/profile/evaluate", {}).status == 404
    again = refused(
        console, "POST", "/api/profile/evaluate", {"adjustment_id": record["adjustment_id"]}
    )
    assert again.status == 422


def test_unknown_adjustment_is_not_found(console, services):
    assert refused(console, "POST", "/api/profile/evaluate", {}).status == 404
    assert refused(
        console, "POST", "/api/profile/evaluate", {"adjustment_id": "adj-999999"}
    ).status == 404


def test_operator_moves_are_checked_against_the_alignment(console, services):
    align(console)
    scan(console, 8.0, 6.0, 8.0)
    record = ok(
        console,
        "POST",
        "/api/profile/adjust",
        {"target": 6, "moves": [{"actuator": "act-front", "move": 25.0}]},
    )
    moves = {move["actuator"]: move["move"] for move in record["moves"]}
    assert moves == {"act-front": 25.0, "act-mid": 0.0, "act-back": 0.0}
    assert refused(
        console,
        "POST",
        "/api/profile/adjust",
        {"target": 6, "moves": [{"actuator": "act-x", "move": 5.0}]},
    ).status == 404
    with pytest.raises(UnknownActuatorError):
        services.profile.adjust(6.0, [{"actuator": "act-x", "move": 5.0}])


def test_target_must_stay_inside_the_moisture_band(console, services):
    align(console)
    scan(console, 8.0, 6.0, 8.0)
    assert refused(console, "GET", "/api/profile/suggest?target=20").status == 422
    assert refused(console, "POST", "/api/profile/adjust", {"target": 1}).status == 422


def test_alignment_and_adjustments_survive_a_restart(config):
    first = build(config)
    align(first.console)
    scan(first.console, 8.0, 6.0, 8.0)
    ok(first.console, "POST", "/api/profile/adjust", {"target": 6})
    scan(first.console, 6.5, 6.0, 6.5)
    ok(first.console, "POST", "/api/profile/evaluate", {})
    second = build(config)
    assert second.profile.generation() == 1
    assert [pair["actuator"] for pair in second.profile.alignment()] == [
        "act-front",
        "act-mid",
        "act-back",
    ]
    (record,) = second.profile.adjustments()
    assert record["status"] == "evaluated"
    assert record["improved"] is True
    scan(second.console, 6.2, 6.0, 6.2)
    follow_up = ok(second.console, "POST", "/api/profile/adjust", {"target": 6})
    assert follow_up["adjustment_id"] == "adj-000002"


def test_state_reports_the_profile_subsystem(console, services):
    align(console)
    scan(console, 8.0, 6.0, 8.0)
    ok(console, "POST", "/api/profile/adjust", {"target": 6})
    state = ok(console, "GET", "/api/profile/state")
    assert state["actuators"] == 3
    assert state["generation"] == 1
    assert state["open_adjustments"] == 1
    assert state["last_improved"] is None
    scan(console, 6.5, 6.0, 6.5)
    ok(console, "POST", "/api/profile/evaluate", {})
    state = ok(console, "GET", "/api/state")
    profile = state["subsystems"]["profile"]
    assert profile["open_adjustments"] == 0
    assert profile["last_improved"] is True

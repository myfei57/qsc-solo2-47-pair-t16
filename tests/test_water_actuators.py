"""Actuator alignment, target profile suggestions and adjustment evaluation."""

import pytest

from paperops.errors import PreGateError, UnknownActuatorError

from support import build, issue_confirmation, ok, refused


def align(console, pairs=None):
    """Build the default one to one alignment against the default sensor map."""
    pairs = pairs or {"zone-front": "front", "zone-middle": "middle", "zone-back": "back"}
    confirmation = issue_confirmation(console, "water.alignment", "water")
    return ok(console, "POST", "/api/water/alignment", {"pairs": pairs, "confirmation_id": confirmation})


def wet_edges(console):
    """A profile with the edges clearly wetter than the middle."""
    ok(console, "POST", "/api/water/sample", {"name": "front", "moisture": 9.0})
    ok(console, "POST", "/api/water/sample", {"name": "middle", "moisture": 6.0})
    ok(console, "POST", "/api/water/sample", {"name": "back", "moisture": 9.0})


def test_alignment_binds_a_confirmation_and_resolves_positions(console, services):
    body = align(console)
    assert body["generation"] == 1
    assert body["map_generation"] == 0
    assert services.water.alignment() == {
        "zone-front": "front",
        "zone-middle": "middle",
        "zone-back": "back",
    }
    state = ok(console, "GET", "/api/water/alignment")
    assert state["positions"] == {"zone-front": 0.4, "zone-middle": 3.1, "zone-back": 5.8}
    assert services.water.view().alignment_generation == 1


def test_alignment_is_one_to_one(console):
    response = refused(
        console,
        "POST",
        "/api/water/alignment",
        {
            "pairs": {"zone-a": "front", "zone-b": "front"},
            "confirmation_id": issue_confirmation(console, "water.alignment", "water"),
        },
    )
    assert response.status == 400
    assert "more than one actuator" in response.body["error"]


def test_alignment_refuses_an_unmapped_sensor(console):
    response = refused(
        console,
        "POST",
        "/api/water/alignment",
        {
            "pairs": {"zone-a": "edge"},
            "confirmation_id": issue_confirmation(console, "water.alignment", "water"),
        },
    )
    assert response.status == 404


def test_suggest_requires_an_alignment(console):
    wet_edges(console)
    response = refused(console, "POST", "/api/water/suggest", {"target": 6.0})
    assert response.status == 404
    with pytest.raises(UnknownActuatorError):
        console.services.water.suggest(target=6.0)


def test_suggest_requires_samples(console):
    align(console)
    response = refused(console, "POST", "/api/water/suggest", {"target": 6.0})
    assert response.status == 409


def test_suggestion_steers_to_target_with_separate_edge_compensation(console, services):
    align(console)
    wet_edges(console)
    body = ok(console, "POST", "/api/water/suggest", {"target": 6.0})
    assert body["generation"] == 1
    moves = {move["actuator"]: move for move in body["moves"]}
    front = moves["zone-front"]
    middle = moves["zone-middle"]
    # the middle is on target and stays; the wet edges move up
    assert middle["error"] == 0.0
    assert middle["edge"] is False
    assert middle["suggested"] == 0.0
    assert front["edge"] is True
    assert front["error"] == 3.0
    # edge compensation is separate from and on top of the error move
    assert body["edge_excess"] == 3.0
    assert front["edge_boost"] == 3.0
    assert front["move"] == 4.0 * 3.0 + 3.0
    assert front["suggested"] == 15.0
    # the frozen before-shape shows the wet edges
    assert body["before"]["edge_center"] == 3.0
    assert body["before"]["mean_abs_error"] == 2.0


def test_suggestion_accepts_a_target_curve(console):
    align(console)
    wet_edges(console)
    body = ok(
        console,
        "POST",
        "/api/water/suggest",
        {"targets": [{"position": 0.4, "moisture": 8.0}, {"position": 5.8, "moisture": 8.0}]},
    )
    moves = {move["actuator"]: move for move in body["moves"]}
    assert moves["zone-front"]["target"] == 8.0
    assert moves["zone-middle"]["target"] == 8.0


def test_apply_sets_the_suggested_setpoints(console, services):
    align(console)
    wet_edges(console)
    ok(console, "POST", "/api/water/suggest", {"target": 6.0})
    adjustment = ok(console, "POST", "/api/water/apply", {})
    assert adjustment["source"] == "suggestion"
    assert adjustment["generation"] == 1
    assert adjustment["setpoints"] == {"zone-front": 15.0, "zone-middle": 0.0, "zone-back": 15.0}
    state = ok(console, "GET", "/api/water/actuators")
    assert state["setpoints"]["zone-front"] == 15.0
    assert state["adjustment"]["before"]["edge_center"] == 3.0


def test_apply_without_a_suggestion_is_a_pre_gate(console):
    align(console)
    wet_edges(console)
    assert refused(console, "POST", "/api/water/apply", {}).status == 409


def test_stale_suggestion_is_refused_after_an_alignment_rebuild(console):
    align(console)
    wet_edges(console)
    ok(console, "POST", "/api/water/suggest", {"target": 6.0})
    align(console)
    response = refused(console, "POST", "/api/water/apply", {})
    assert response.status == 409


def test_sensor_map_rebuild_invalidates_the_alignment(console):
    align(console)
    wet_edges(console)
    ok(
        console,
        "POST",
        "/api/water/map",
        {
            "sensors": {"front": 0.5, "middle": 3.0, "back": 5.5},
            "confirmation_id": issue_confirmation(console, "water.map", "water"),
        },
    )
    assert refused(console, "POST", "/api/water/suggest", {"target": 6.0}).status == 409
    with pytest.raises(PreGateError):
        console.services.water.evaluate()


def test_manual_moves_are_checked_against_the_alignment(console, services):
    align(console)
    wet_edges(console)
    response = refused(console, "POST", "/api/water/actuators", {"setpoints": {"zone-x": 40}})
    assert response.status == 404
    response = refused(console, "POST", "/api/water/actuators", {"setpoints": {"zone-front": 140}})
    assert response.status == 422
    adjustment = ok(
        console, "POST", "/api/water/actuators", {"setpoints": {"zone-front": 40}, "target": 6.0}
    )
    assert adjustment["source"] == "manual"
    assert services.water.setpoints()["zone-front"] == 40.0


def test_evaluation_shows_the_profile_improved(console, services):
    align(console)
    wet_edges(console)
    ok(console, "POST", "/api/water/suggest", {"target": 6.0})
    ok(console, "POST", "/api/water/apply", {})
    # the scanner then sees the edges come down towards the target
    ok(console, "POST", "/api/water/sample", {"name": "front", "moisture": 6.5})
    ok(console, "POST", "/api/water/sample", {"name": "middle", "moisture": 6.0})
    ok(console, "POST", "/api/water/sample", {"name": "back", "moisture": 6.5})
    body = ok(console, "POST", "/api/water/evaluate", {})
    assert body["verdict"] == "improved"
    assert body["generation"] == 1
    assert body["before"]["mean_abs_error"] == 2.0
    assert body["after"]["mean_abs_error"] == pytest.approx(1.0 / 3.0)
    assert body["mean_abs_error_delta"] < 0
    assert body["span_delta"] == -2.5
    assert body["edge_center_delta"] == -2.5
    history = services.water.adjustment_history().to_dict()
    assert history["current"] == "improved"
    listing = ok(console, "GET", "/api/water/evaluations")
    assert listing["latest"]["verdict"] == "improved"
    assert services.state.histories()[-2] == "water.adjustment"


def test_evaluation_calls_a_worsening_move_out(console):
    align(console)
    wet_edges(console)
    ok(console, "POST", "/api/water/actuators", {"setpoints": {"zone-middle": 30}, "target": 6.0})
    ok(console, "POST", "/api/water/sample", {"name": "front", "moisture": 10.0})
    ok(console, "POST", "/api/water/sample", {"name": "middle", "moisture": 6.0})
    ok(console, "POST", "/api/water/sample", {"name": "back", "moisture": 10.0})
    body = ok(console, "POST", "/api/water/evaluate", {})
    assert body["verdict"] == "worsened"
    assert body["mean_abs_error_delta"] > 0


def test_evaluate_without_an_adjustment_is_a_pre_gate(console):
    align(console)
    wet_edges(console)
    assert refused(console, "POST", "/api/water/evaluate", {}).status == 409


def test_alignment_and_setpoints_survive_a_restart(config):
    first = build(config)
    align(first.console)
    wet_edges(first.console)
    ok(first.console, "POST", "/api/water/suggest", {"target": 6.0})
    ok(first.console, "POST", "/api/water/apply", {})
    ok(first.console, "POST", "/api/water/sample", {"name": "front", "moisture": 6.5})
    ok(first.console, "POST", "/api/water/evaluate", {})
    second = build(config)
    assert second.water.alignment()["zone-front"] == "front"
    assert second.water.setpoints()["zone-front"] == 15.0
    assert second.water.alignment_generation() == 1
    evaluations = second.water.evaluations()
    assert len(evaluations) == 1
    assert evaluations[0]["generation"] == 1

"""Step 4: conditional routing after validation and the repair loop."""

import pytest
from helpers import make_plan, make_req, make_task

from flowsmith import graph as graph_module
from flowsmith.graph import MAX_REPAIR_ATTEMPTS, build_graph, route_after_validate
from flowsmith.prompts import REPAIR_SYSTEM
from flowsmith.schemas import Plan, Requirements
from flowsmith.validation import Issue, ValidationReport, validate_plan

CONFIG = {"recursion_limit": 20}


# ---------------------------------------------------------------------------
# Fixtures and builders
# ---------------------------------------------------------------------------

def good_plan():
    return make_plan(make_task("T01"), make_task("T02", depends_on=["T01"]))


def cyclic_plan():
    return make_plan(
        make_task("T01", depends_on=["T02"]),
        make_task("T02", depends_on=["T01"]),
    )


def missing_dep_plan():
    return make_plan(make_task("T01", depends_on=["T99"]))


def warnings_only_plan():
    # 'must' T02 depends on 'could' T01 -> priority_inversion warning, no errors.
    return make_plan(
        make_task("T01", priority="could"),
        make_task("T02", depends_on=["T01"]),
    )


class FakeLLM:
    """Stands in for call_structured: Requirements are fixed, plans come from a script."""

    def __init__(self, plans, req=None):
        self.plans = list(plans)
        self.req = req or make_req()
        self.calls = []  # (schema, system, user)

    def __call__(self, llm, schema, system, user):
        self.calls.append((schema, system, user))
        if schema is Requirements:
            return self.req
        if not self.plans:
            raise AssertionError("FakeLLM ran out of scripted plans (unexpected extra call)")
        return self.plans.pop(0)

    @property
    def repair_calls(self):
        return [c for c in self.calls if c[1] == REPAIR_SYSTEM]


@pytest.fixture
def run(monkeypatch):
    """run(plans) -> (final_state, fake). Plans are returned in order by decompose/repair."""

    def _run(plans, req=None):
        fake = FakeLLM(plans, req)
        monkeypatch.setattr(graph_module, "call_structured", fake)
        result = build_graph(llm=None).invoke({"idea": "a todo app"}, config=CONFIG)
        return result, fake

    return _run


def report(*levels):
    return ValidationReport(
        issues=[Issue(code="x", level=lvl, message=f"{lvl} issue") for lvl in levels]
    )


# ---------------------------------------------------------------------------
# 1. route_after_validate in isolation
# ---------------------------------------------------------------------------

def test_route_valid_report_goes_to_schedule():
    # Bug: is_valid check inverted, or a valid report is sent to "repair".
    assert route_after_validate({"validation": report()}) == "schedule"


def test_route_invalid_report_without_attempts_key_goes_to_repair():
    # Bug: state["repair_attempts"] used instead of .get(...) -> KeyError on first failure.
    assert route_after_validate({"validation": report("error")}) == "repair"


def test_route_invalid_report_below_max_attempts_goes_to_repair():
    # Bug: comparison uses <= MAX-1 / wrong constant, so it gives up while attempts remain.
    state = {"validation": report("error"), "repair_attempts": MAX_REPAIR_ATTEMPTS - 1}
    assert route_after_validate(state) == "repair"


def test_route_invalid_report_at_max_attempts_gives_up():
    # Bug: `<=` instead of `<` -> one repair too many.
    state = {"validation": report("error"), "repair_attempts": MAX_REPAIR_ATTEMPTS}
    assert route_after_validate(state) == "give_up"


def test_route_invalid_report_above_max_attempts_gives_up():
    # Bug: `== MAX` used for giving up -> an overshooting counter loops forever.
    state = {"validation": report("error"), "repair_attempts": MAX_REPAIR_ATTEMPTS + 1}
    assert route_after_validate(state) == "give_up"


def test_route_warnings_only_report_goes_to_schedule():
    # Bug: routing on `report.issues` instead of `report.errors` repairs harmless warnings.
    assert route_after_validate({"validation": report("warning")}) == "schedule"


def test_route_valid_report_goes_to_schedule_even_at_max_attempts():
    # Bug: attempt limit checked before validity -> a fixed plan is thrown away.
    state = {"validation": report(), "repair_attempts": MAX_REPAIR_ATTEMPTS}
    assert route_after_validate(state) == "schedule"


# ---------------------------------------------------------------------------
# 2. Full graph with a fake LLM
# ---------------------------------------------------------------------------

def test_valid_first_plan_schedules_without_repair(run):
    # Bug: validate always routes to repair, or repair runs unconditionally.
    result, fake = run([good_plan()])
    assert fake.repair_calls == []
    assert result.get("repair_attempts", 0) == 0
    assert "schedule" in result


def test_valid_first_plan_makes_exactly_two_llm_calls(run):
    # Bug: an extra LLM call sneaks in (e.g. repair called on a valid plan, or analyze twice).
    _, fake = run([good_plan()])
    assert [c[0] for c in fake.calls] == [Requirements, Plan]


def test_bad_then_good_plan_repairs_once_and_schedules(run):
    # Bug: repair edge doesn't loop back to validate, or attempts aren't counted.
    result, fake = run([cyclic_plan(), good_plan()])
    assert len(fake.repair_calls) == 1
    assert result["repair_attempts"] == 1
    assert result["validation"].is_valid
    assert "schedule" in result


def test_repaired_plan_replaces_original_in_final_state(run):
    # Bug: repair returns its plan under the wrong key, so state keeps the broken plan.
    fixed = good_plan()
    result, _ = run([cyclic_plan(), fixed])
    assert result["plan"] == fixed


def test_always_bad_stops_at_max_repairs_without_schedule(run):
    # Bug: counter never incremented / give_up unmapped -> loop until recursion limit error.
    result, fake = run([cyclic_plan()] * (MAX_REPAIR_ATTEMPTS + 1))
    assert len(fake.repair_calls) == MAX_REPAIR_ATTEMPTS
    assert result["repair_attempts"] == MAX_REPAIR_ATTEMPTS
    assert "schedule" not in result


def test_always_bad_leaves_failing_validation_in_state(run):
    # Bug: give_up path drops/overwrites the report, so the caller can't say why it failed.
    result, _ = run([cyclic_plan()] * (MAX_REPAIR_ATTEMPTS + 1))
    assert not result["validation"].is_valid
    assert any("Circular dependency" in e.message for e in result["validation"].errors)


def test_fix_on_last_allowed_repair_is_accepted(run):
    # Bug: off-by-one (counter starts at 1 or `< MAX-1`) gives up one repair early.
    plans = [cyclic_plan()] * MAX_REPAIR_ATTEMPTS + [good_plan()]
    result, fake = run(plans)
    assert len(fake.repair_calls) == MAX_REPAIR_ATTEMPTS
    assert result["validation"].is_valid
    assert "schedule" in result


def test_warnings_only_plan_is_scheduled_without_repair(run):
    # Bug: graph treats any issue (incl. warnings) as failure and starts repairing.
    result, fake = run([warnings_only_plan()])
    assert result["validation"].warnings  # precondition: there really are warnings
    assert fake.repair_calls == []
    assert "schedule" in result


# --- the repair call itself ------------------------------------------------

def test_repair_message_contains_exact_validator_errors(run):
    # Bug: repair sends a generic "fix the plan" message or only error codes.
    bad = cyclic_plan()
    expected = [e.message for e in validate_plan(bad, make_req()).errors]
    _, fake = run([bad, good_plan()])
    user = fake.repair_calls[0][2]
    assert "Circular dependency" in user
    for msg in expected:
        assert msg in user


def test_repair_message_contains_current_plan_json(run):
    # Bug: repair omits the plan, so the LLM has to regenerate from scratch.
    bad = cyclic_plan()
    _, fake = run([bad, good_plan()])
    user = fake.repair_calls[0][2]
    assert bad.model_dump_json(indent=2) in user or bad.model_dump_json() in user


def test_repair_message_contains_requirements_json(run):
    # Bug: requirements dropped -> LLM can't fix "feature not covered" errors.
    req = make_req("F1")
    _, fake = run([cyclic_plan(), good_plan()], req=req)
    user = fake.repair_calls[0][2]
    assert req.model_dump_json(indent=2) in user or req.model_dump_json() in user


def test_repair_uses_repair_system_prompt(run):
    # Bug: repair reuses DECOMPOSE_SYSTEM, so no call is made with REPAIR_SYSTEM.
    _, fake = run([cyclic_plan(), good_plan()])
    systems = [c[1] for c in fake.calls]
    assert systems.count(REPAIR_SYSTEM) == 1
    assert systems[2] == REPAIR_SYSTEM  # third call: after analyze and decompose


def test_repair_requests_plan_schema(run):
    # Bug: repair passes Requirements (copy-paste) as the schema.
    _, fake = run([cyclic_plan(), good_plan()])
    assert fake.repair_calls[0][0] is Plan


@pytest.mark.skipif(MAX_REPAIR_ATTEMPTS < 2, reason="needs at least two repairs")
def test_second_repair_sees_latest_plan_and_latest_errors(run):
    # Bug: repair reads a stale plan/validation (e.g. captured from the first round).
    first_bad, second_bad = cyclic_plan(), missing_dep_plan()
    _, fake = run([first_bad, second_bad, good_plan()])
    second_user = fake.repair_calls[1][2]
    assert "T01 depends on T99, which does not exist." in second_user
    assert "Circular dependency" not in second_user
    assert second_bad.model_dump_json(indent=2) in second_user or (
        second_bad.model_dump_json() in second_user
    )


# --- log -------------------------------------------------------------------

def test_log_has_one_entry_per_node_in_order_after_one_repair(run):
    # Bug: log reducer missing (entries overwrite) or a node forgets to log.
    result, _ = run([cyclic_plan(), good_plan()])
    nodes = [entry.split(":", 1)[0] for entry in result["log"]]
    assert nodes == ["analyze", "decompose", "validate", "repair", "validate", "schedule"]


def test_log_matches_nodes_that_ran_when_giving_up(monkeypatch):
    # Bug: a node logs twice/never, or give_up path appends a stray entry without a node.
    fake = FakeLLM([cyclic_plan()] * (MAX_REPAIR_ATTEMPTS + 1))
    monkeypatch.setattr(graph_module, "call_structured", fake)
    app = build_graph(llm=None)

    ran, state = [], None
    for mode, chunk in app.stream(
        {"idea": "a todo app"}, config=CONFIG, stream_mode=["updates", "values"]
    ):
        if mode == "updates":
            ran += list(chunk)
        else:
            state = chunk

    assert [e.split(":", 1)[0] for e in state["log"]] == ran
    assert ran == ["analyze", "decompose"] + ["validate", "repair"] * MAX_REPAIR_ATTEMPTS + [
        "validate"
    ]

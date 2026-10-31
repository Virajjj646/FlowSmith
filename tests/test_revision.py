"""Step 6b: plan review, the revision agent loop, discard and rollback."""

import logging

import pytest
from helpers import (
    ScriptedAgent,
    ScriptedLLM,
    agent_reply,
    make_multi_module_plan,
    make_plan,
    make_req,
    make_task,
    make_valid_plan,
    new_task_args,
    tool_calls,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.serde import jsonplus
from langgraph.types import Command

from flowsmith import graph as graph_module
from flowsmith.checkpoint import make_checkpointer
from flowsmith.graph import (
    MAX_AGENT_STEPS,
    MAX_REPAIR_ATTEMPTS,
    build_graph,
    build_revision_message,
    plan_overview,
    route_after_agent,
    route_after_review,
    route_after_validate,
    valid_decision,
)
from flowsmith.prompts import REPAIR_SYSTEM, REVISE_SYSTEM
from flowsmith.tools import TOOLS
from flowsmith.validation import Issue, ValidationReport

IDEA = "a todo app"
REQ = make_req("F1", "F2")
SERDE_LOGGER = "langgraph.checkpoint.serde.jsonplus"
APPROVE = {"action": "approve"}
DISCARD = {"action": "discard"}


def change(request):
    return {"action": "change", "request": request}


def cyclic_plan():
    """make_multi_module_plan with T01 -> T02 -> T01."""
    plan = make_multi_module_plan()
    tasks = [t.model_copy(update={"depends_on": ["T02"]}) if t.id == "T01" else t for t in plan.tasks]
    return plan.model_copy(update={"tasks": tasks})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(request):
    return {"configurable": {"thread_id": request.node.nodeid}, "recursion_limit": 60}


@pytest.fixture
def start(monkeypatch, cfg):
    """start(*agent_responses, repair_plans=None, plan=None, req=REQ) -> (app, agent, llm, first_result).

    Runs the graph to the first review pause (plan v1 = make_multi_module_plan()).
    """

    def _start(*responses, repair_plans=None, plan=None, req=REQ):
        llm = ScriptedLLM(req, plan=plan or make_multi_module_plan(), repair_plans=repair_plans)
        agent = ScriptedAgent(*responses)
        monkeypatch.setattr(graph_module, "call_structured", llm)
        monkeypatch.setattr(graph_module, "call_agent", agent)
        app = build_graph(llm=None, checkpointer=make_checkpointer(), human_review=True)
        result = app.invoke({"idea": IDEA}, cfg)
        return app, agent, llm, result

    return _start


def payload(result):
    [interrupt] = result["__interrupt__"]
    return interrupt.value


def without_notice(p):
    return {k: v for k, v in p.items() if k != "notice"}


def resume(app, cfg, decision):
    return app.invoke(Command(resume=decision), cfg)


def tool_messages(app, cfg):
    return [m for m in app.get_state(cfg).values["messages"] if isinstance(m, ToolMessage)]


def ids(plan):
    return [t.id for t in plan.tasks]


# ---------------------------------------------------------------------------
# 7. First review
# ---------------------------------------------------------------------------

def test_first_pause_is_review_of_v1_without_diff(start, cfg):
    # Bug: review not wired after schedule, or v1 offered a diff/discard against nothing.
    app, _, _, result = start()
    assert app.get_state(cfg).next == ("review",)
    assert payload(result) == {
        "kind": "review", "plan_version": 1, "diff": None, "can_discard": False, "notice": None,
    }


# ---------------------------------------------------------------------------
# 8. Approve
# ---------------------------------------------------------------------------

def test_approve_ends_the_run(start, cfg):
    # Bug: approve routed back into revise/review instead of END.
    app, agent, _, _ = start()
    result = resume(app, cfg, APPROVE)
    assert app.get_state(cfg).next == ()
    assert "__interrupt__" not in result
    assert agent.calls == []


# ---------------------------------------------------------------------------
# 9. Change request
# ---------------------------------------------------------------------------

def test_change_request_calls_agent_with_revise_system_and_tools(start, cfg):
    # Bug: revise reuses DECOMPOSE/REPAIR prompt or binds no tools, so the model can't edit.
    app, agent, _, _ = start(agent_reply())
    resume(app, cfg, change("Make T03 large"))
    system, _, tools = agent.calls[0]
    assert system == REVISE_SYSTEM
    assert tools == TOOLS


def test_first_agent_call_gets_one_human_message_with_request_and_every_task_id(start, cfg):
    # Bug: request or plan overview missing from the message, so the model edits blind.
    app, agent, _, _ = start(agent_reply())
    resume(app, cfg, change("  Make T03 large  "))
    _, messages, _ = agent.calls[0]
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    content = messages[0].content
    assert "Change request: Make T03 large\n" in content  # stripped
    for task_id in ("T01", "T02", "T03", "T04"):
        assert f"\n{task_id} | " in content


def test_tool_message_ids_match_their_calls(start, cfg):
    # Bug: ToolMessages reuse one id or are misordered -> provider rejects the conversation.
    first = tool_calls(("GetTask", {"task_id": "T03"}), ("UpdateTask", {"task_id": "T03", "changes": {"size": "L"}}))
    app, _, _, _ = start(first, agent_reply())
    resume(app, cfg, change("Make T03 large"))
    expected = [c["id"] for c in first.tool_calls]
    assert [m.tool_call_id for m in tool_messages(app, cfg)] == expected


def test_agent_sees_tool_results_on_its_next_step(start, cfg):
    # Bug: tools node results not appended to messages, so the model never sees what happened.
    app, agent, _, _ = start(tool_calls(("GetTask", {"task_id": "T03"})), agent_reply())
    resume(app, cfg, change("Make T03 large"))
    _, second_messages, _ = agent.calls[1]
    assert [type(m) for m in second_messages] == [HumanMessage, AIMessage, ToolMessage]
    assert '"id":"T03"' in second_messages[2].content


def test_next_review_is_v2_with_discard_and_correct_diff(start, cfg):
    # Bug: version not bumped, base_plan not saved, or diff computed against the wrong plan.
    app, _, _, _ = start(
        tool_calls(("UpdateTask", {"task_id": "T03", "changes": {"size": "L"}})), agent_reply()
    )
    result = resume(app, cfg, change("Make T03 large"))
    assert payload(result) == {
        "kind": "review",
        "plan_version": 2,
        "diff": {"added": [], "removed": [], "changed": [{"id": "T03", "fields": ["size"]}],
                 "unchanged": 3, "added_modules": []},
        "can_discard": True,
        "notice": None,
    }


# ---------------------------------------------------------------------------
# 10. Tool errors
# ---------------------------------------------------------------------------

def test_tool_error_becomes_error_tool_message(start, cfg):
    # Bug: PlanEditError escapes the tools node and crashes the run instead of reaching the model.
    app, _, _, _ = start(tool_calls(("RemoveTask", {"task_id": "T01"})), agent_reply())
    resume(app, cfg, change("Remove T01"))
    [message] = tool_messages(app, cfg)
    assert message.content.startswith("ERROR: Cannot remove T01: T02, T03 depend on it.")


def test_failed_call_leaves_plan_unchanged_while_others_in_batch_apply(start, cfg):
    # Bug: an error aborts the whole batch, or the failed call's partial result is kept.
    batch = tool_calls(
        ("RemoveTask", {"task_id": "T01"}),
        ("UpdateTask", {"task_id": "T03", "changes": {"size": "L"}}),
    )
    app, _, _, _ = start(batch, agent_reply())
    result = resume(app, cfg, change("Resize T03"))
    diff = payload(result)["diff"]
    assert (diff["removed"], diff["changed"]) == ([], [{"id": "T03", "fields": ["size"]}])


def test_loop_continues_after_error_so_a_corrected_call_succeeds(start, cfg):
    # Bug: route_after_agent/tools stops the loop on error, so the model can't retry.
    app, agent, _, _ = start(
        tool_calls(("UpdateTask", {"task_id": "T99", "changes": {"size": "L"}})),
        tool_calls(("UpdateTask", {"task_id": "T03", "changes": {"size": "L"}})),
        agent_reply(),
    )
    result = resume(app, cfg, change("Make T03 large"))
    assert len(agent.calls) == 3
    assert [m.content.startswith("ERROR:") for m in tool_messages(app, cfg)] == [True, False]
    assert payload(result)["diff"]["changed"] == [{"id": "T03", "fields": ["size"]}]


# ---------------------------------------------------------------------------
# 11. Step limit
# ---------------------------------------------------------------------------

def test_agent_is_called_at_most_max_steps_then_proceeds_to_review(start, cfg):
    # Bug: steps counter never incremented -> the agent loops until the recursion limit.
    responses = [tool_calls(("GetTask", {"task_id": "T01"})) for _ in range(MAX_AGENT_STEPS + 2)]
    app, agent, _, _ = start(*responses)
    result = resume(app, cfg, change("Keep thinking"))
    assert len(agent.calls) == MAX_AGENT_STEPS
    assert app.get_state(cfg).next == ("review",)
    assert payload(result)["plan_version"] == 2


# ---------------------------------------------------------------------------
# 12. Discard
# ---------------------------------------------------------------------------

def discard_after_one_revision(start, cfg):
    app, _, _, _ = start(tool_calls(("RemoveTask", {"task_id": "T04"})), agent_reply())
    resume(app, cfg, change("Drop T04"))
    return app, resume(app, cfg, DISCARD)


def test_discard_restores_previous_plan(start, cfg):
    # Bug: rollback restores nothing (plan stays v2) or restores the wrong snapshot.
    app, result = discard_after_one_revision(start, cfg)
    assert result["plan"] == make_multi_module_plan()


def test_discard_restores_previous_version_and_disables_discard(start, cfg):
    # Bug: plan_version not decremented or base_plan not cleared -> discard offered on v1.
    app, result = discard_after_one_revision(start, cfg)
    assert without_notice(payload(result)) == {"kind": "review", "plan_version": 1, "diff": None, "can_discard": False}


def test_discard_logs_discard_then_rollback(start, cfg):
    # Bug: discard routed somewhere other than rollback.
    app, result = discard_after_one_revision(start, cfg)
    log = result["log"]
    i = log.index("review: discarded v2")
    assert log[i + 1 : i + 3] == ["rollback: restored v1", "validate: 0 errors, 0 warnings"]


# ---------------------------------------------------------------------------
# 13. Rollback after failed repair
# ---------------------------------------------------------------------------

def run_cycle_revision(start, cfg):
    app, agent, llm, _ = start(
        tool_calls(("UpdateTask", {"task_id": "T01", "changes": {"depends_on": ["T02"]}})),
        agent_reply("Made T01 wait for T02."),
        repair_plans=[cyclic_plan()] * MAX_REPAIR_ATTEMPTS,
    )
    result = resume(app, cfg, change("T01 should wait for T02"))
    return app, llm, result


def test_failed_revision_is_repaired_max_times_before_rollback(start, cfg):
    # Bug: repair_attempts not reset on change request -> a revision gets fewer (or no) repairs.
    app, llm, result = run_cycle_revision(start, cfg)
    assert sum(entry.startswith("repair:") for entry in result["log"]) == MAX_REPAIR_ATTEMPTS
    assert llm.repair_plans == []


def test_failed_revision_rolls_back_to_previous_plan(start, cfg):
    # Bug: give_up taken instead of rollback, ending the run with an invalid plan.
    app, _, result = run_cycle_revision(start, cfg)
    assert "rollback: restored v1" in result["log"]
    assert result["plan"] == make_multi_module_plan()
    assert result["validation"].is_valid


def test_rollback_pauses_at_review_of_restored_version(start, cfg):
    # Bug: rollback edge goes to END/schedule without review, so the user never sees the result.
    app, _, result = run_cycle_revision(start, cfg)
    assert app.get_state(cfg).next == ("review",)
    assert without_notice(payload(result)) == {"kind": "review", "plan_version": 1, "diff": None, "can_discard": False}


# ---------------------------------------------------------------------------
# 14. Second revision starts fresh
# ---------------------------------------------------------------------------

def test_second_revision_starts_with_one_human_message(start, cfg):
    # Bug: REMOVE_ALL_MESSAGES missing -> the agent replays the first revision's conversation.
    app, agent, _, _ = start(
        tool_calls(("UpdateTask", {"task_id": "T03", "changes": {"size": "L"}})), agent_reply(),
        agent_reply(),
    )
    resume(app, cfg, change("Make T03 large"))
    resume(app, cfg, change("Rename T02"))
    _, messages, _ = agent.calls[2]
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert "Change request: Rename T02" in messages[0].content


def test_second_revision_overview_shows_the_revised_plan(start, cfg):
    # Bug: revision message built from base_plan (stale) instead of the current plan.
    app, agent, _, _ = start(
        tool_calls(("UpdateTask", {"task_id": "T03", "changes": {"size": "L"}})), agent_reply(),
        agent_reply(),
    )
    resume(app, cfg, change("Make T03 large"))
    resume(app, cfg, change("Anything else"))
    content = agent.calls[2][1][0].content
    assert "T03 | Core | L |" in content


def test_second_revision_resets_agent_steps_and_repair_attempts(start, cfg):
    # Bug: counters carried over -> the second revision hits the step/repair limit early.
    app, _, _, _ = start(
        tool_calls(("GetTask", {"task_id": "T01"})), tool_calls(("GetTask", {"task_id": "T01"})), agent_reply(),
        agent_reply(),
    )
    resume(app, cfg, change("First"))
    assert app.get_state(cfg).values["agent_steps"] == 3
    resume(app, cfg, change("Second"))
    values = app.get_state(cfg).values
    assert values["agent_steps"] == 1
    assert values["repair_attempts"] == 0


def test_second_revision_diff_is_against_v2(start, cfg):
    # Bug: base_plan not updated on the second change, so v3's diff is against v1.
    app, _, _, _ = start(
        tool_calls(("UpdateTask", {"task_id": "T03", "changes": {"size": "L"}})), agent_reply(),
        tool_calls(("UpdateTask", {"task_id": "T02", "changes": {"title": "Login"}})), agent_reply(),
    )
    resume(app, cfg, change("Make T03 large"))
    result = resume(app, cfg, change("Rename T02"))
    p = payload(result)
    assert p["plan_version"] == 3
    assert p["diff"]["changed"] == [{"id": "T02", "fields": ["title"]}]


# ---------------------------------------------------------------------------
# 15. Ids are not reused
# ---------------------------------------------------------------------------

def test_removing_highest_id_then_adding_gives_new_id(start, cfg):
    # Bug: tools node doesn't pass base ids as reserved_ids -> the new task becomes "T04" again.
    app, _, _, _ = start(
        tool_calls(("RemoveTask", {"task_id": "T04"})),
        tool_calls(("AddTask", new_task_args(module="Auth", depends_on=["T02"], covers=["F2"]))),
        agent_reply(),
    )
    result = resume(app, cfg, change("Replace T04"))
    assert [m.content for m in tool_messages(app, cfg)] == ["Removed T04.", "Added T05."]
    diff = payload(result)["diff"]
    assert (diff["removed"], diff["added"], diff["changed"]) == (["T04"], ["T05"], [])


# ---------------------------------------------------------------------------
# 16. Invalid decisions are asked again
# ---------------------------------------------------------------------------

INVALID_AT_V1 = {
    "non-dict": "approve",
    "unknown action": {"action": "ship-it"},
    "empty request": {"action": "change", "request": "   "},
    "request not str": {"action": "change", "request": 42},
    "discard not allowed on v1": DISCARD,
}


@pytest.mark.parametrize("decision", INVALID_AT_V1.values(), ids=INVALID_AT_V1.keys())
def test_invalid_decision_is_asked_again_with_error(start, cfg, decision):
    # Bug: invalid decision raises (KeyError/AttributeError) and leaves the thread stuck.
    app, agent, _, _ = start()
    result = resume(app, cfg, decision)
    p = payload(result)
    assert app.get_state(cfg).next == ("review",)
    assert "error" in p
    assert (p["kind"], p["plan_version"], p["can_discard"]) == ("review", 1, False)
    assert agent.calls == []


def test_error_on_v1_does_not_offer_discard(start, cfg):
    # Bug: error text always lists discard, inviting an option that will be rejected again.
    app, _, _, _ = start()
    p = payload(resume(app, cfg, "nonsense"))
    assert "discard" not in p["error"]


def test_error_on_v2_offers_discard(start, cfg):
    # Bug: discard missing from the error hint when it is actually allowed.
    app, _, _, _ = start(agent_reply())
    resume(app, cfg, change("Anything"))
    p = payload(resume(app, cfg, "nonsense"))
    assert "{'action': 'discard'}" in p["error"]


def test_valid_decision_after_several_invalid_ones_works(start, cfg):
    # Bug: `if` instead of `while` -> the second invalid decision is accepted (or crashes).
    app, _, _, _ = start()
    for bad in INVALID_AT_V1.values():
        resume(app, cfg, bad)
    result = resume(app, cfg, APPROVE)
    assert app.get_state(cfg).next == ()
    assert "__interrupt__" not in result


# ---------------------------------------------------------------------------
# 17. human_review=False
# ---------------------------------------------------------------------------

def test_without_human_review_graph_has_no_review_nodes():
    # Bug: review nodes registered unconditionally, so batch runs pause for a human.
    nodes = set(build_graph(None).get_graph().nodes)
    assert nodes.isdisjoint({"review", "revise", "tools", "rollback"})


def test_without_human_review_run_ends_after_schedule(monkeypatch):
    # Bug: schedule wired to review even when human_review is off.
    monkeypatch.setattr(graph_module, "call_structured", ScriptedLLM(REQ, plan=make_multi_module_plan()))
    result = build_graph(None).invoke({"idea": IDEA}, {"recursion_limit": 60})
    assert "__interrupt__" not in result
    assert "schedule" in result
    assert result["log"][-1].startswith("schedule:")


def test_with_human_review_graph_has_review_nodes():
    # Bug: human_review flag ignored, so the CLI never gets a review step.
    nodes = set(build_graph(None, human_review=True).get_graph().nodes)
    assert {"review", "revise", "tools", "rollback"} <= nodes


# ---------------------------------------------------------------------------
# 18. Serializer warnings
# ---------------------------------------------------------------------------

def test_full_revision_cycle_logs_no_unregistered_type_warning(start, cfg, monkeypatch, caplog):
    # Bug: a new state type (e.g. a message class or PlanDiff) is checkpointed without being allowed.
    monkeypatch.setattr(jsonplus, "_warned_unregistered_types", set())
    caplog.set_level(logging.WARNING, logger=SERDE_LOGGER)
    app, _, _, _ = start(
        tool_calls(("RemoveTask", {"task_id": "T04"}), ("AddTask", new_task_args(covers=["F2"]))),
        agent_reply(),
    )
    resume(app, cfg, change("Replace T04"))
    resume(app, cfg, DISCARD)
    resume(app, cfg, APPROVE)
    app.get_state(cfg)
    assert [r.getMessage() for r in caplog.records if "unregistered type" in r.getMessage()] == []


# ---------------------------------------------------------------------------
# 19. Unit tests for the pure functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "action, route", [("approve", "approve"), ("change", "revise"), ("discard", "rollback")]
)
def test_route_after_review_maps_actions(action, route):
    # Bug: change/discard routes swapped, so a discard starts a new revision.
    assert route_after_review({"review_action": action}) == route


def test_route_after_agent_with_tool_calls_below_limit_goes_to_tools():
    # Bug: tool calls ignored, so the agent's edits are never applied.
    state = {"messages": [tool_calls(("GetTask", {"task_id": "T01"}))], "agent_steps": MAX_AGENT_STEPS - 1}
    assert route_after_agent(state) == "tools"


def test_route_after_agent_at_limit_goes_to_validate():
    # Bug: `<=` instead of `<` -> one tool round more than MAX_AGENT_STEPS allows.
    state = {"messages": [tool_calls(("GetTask", {"task_id": "T01"}))], "agent_steps": MAX_AGENT_STEPS}
    assert route_after_agent(state) == "validate"


def test_route_after_agent_without_tool_calls_goes_to_validate():
    # Bug: final answer treated as a tool request -> tools node runs with nothing to do.
    assert route_after_agent({"messages": [agent_reply()], "agent_steps": 1}) == "validate"


def report(*levels):
    return ValidationReport(issues=[Issue(code="x", level=lvl, message="m") for lvl in levels])


def test_route_after_validate_rolls_back_revision_when_repairs_exhausted():
    # Bug: base_plan check missing -> a failed revision ends the run instead of restoring.
    state = {"validation": report("error"), "repair_attempts": MAX_REPAIR_ATTEMPTS, "base_plan": make_plan()}
    assert route_after_validate(state) == "rollback"


def test_route_after_validate_gives_up_without_base_plan():
    # Bug: rollback attempted on the initial plan, where there is nothing to restore.
    state = {"validation": report("error"), "repair_attempts": MAX_REPAIR_ATTEMPTS, "base_plan": None}
    assert route_after_validate(state) == "give_up"


def test_route_after_validate_repairs_revision_before_rolling_back():
    # Bug: rollback checked before repair, so a fixable revision is thrown away.
    state = {"validation": report("error"), "repair_attempts": 0, "base_plan": make_plan()}
    assert route_after_validate(state) == "repair"


@pytest.mark.parametrize(
    "decision, can_discard, expected",
    [
        ({"action": "approve"}, False, True),
        ({"action": "discard"}, True, True),
        ({"action": "discard"}, False, False),
        ({"action": "change", "request": "do x"}, False, True),
        ({"action": "change", "request": " \t"}, False, False),
        ({"action": "change"}, False, False),
        ({"action": "change", "request": None}, False, False),
        ({"action": "APPROVE"}, False, False),
        ({}, False, False),
        (["approve"], False, False),
        (None, False, False),
    ],
)
def test_valid_decision(decision, can_discard, expected):
    # Bug: a malformed decision passes validation and crashes review (e.g. None.strip()).
    assert valid_decision(decision, can_discard) is expected


def test_plan_overview_format():
    # Bug: overview drops deps/covers or uses a different column order than the header promises.
    plan = make_plan(make_task("T01", covers=()), make_task("T02", depends_on=["T01"], size="M", priority="could"))
    assert plan_overview(plan) == (
        "Modules: Core\n"
        "T01 | Core | S | must | after: - | covers: - | Task T01\n"
        "T02 | Core | M | could | after: T01 | covers: F1 | Task T02"
    )


def test_build_revision_message_contains_request_requirements_and_overview():
    # Bug: requirements omitted, so the agent can't set 'covers' correctly.
    plan = make_multi_module_plan()
    message = build_revision_message({"requirements": REQ, "plan": plan}, "Add billing")
    assert message.startswith("Change request: Add billing\n\nRequirements:\n")
    assert REQ.model_dump_json(indent=2) in message
    assert message.endswith(plan_overview(plan))


# ---------------------------------------------------------------------------
# Step 6b fixes: used_ids, retired ids in repair, step-limit log, notices
# ---------------------------------------------------------------------------

def test_id_added_then_removed_in_same_revision_is_not_reused(start, cfg):
    # Bug: only base-plan ids are reserved -> the second AddTask is "T03" again.
    app, _, _, _ = start(
        tool_calls(("AddTask", new_task_args())),
        tool_calls(("RemoveTask", {"task_id": "T03"})),
        tool_calls(("AddTask", new_task_args())),
        agent_reply(),
        plan=make_valid_plan(), req=make_req(),
    )
    resume(app, cfg, change("Add, drop, add"))
    assert [m.content for m in tool_messages(app, cfg)] == ["Added T03.", "Removed T03.", "Added T04."]


def test_used_ids_reset_at_start_of_each_revision(start, cfg):
    # Bug: used_ids carried over between revisions, so ids that never reached an approved plan stay burnt.
    app, _, _, _ = start(
        tool_calls(("AddTask", new_task_args(covers=["F2"]))),
        tool_calls(("RemoveTask", {"task_id": "T05"})),
        agent_reply(),
        agent_reply(),
    )
    resume(app, cfg, change("Add and drop T05"))
    assert "T05" in app.get_state(cfg).values["used_ids"]
    resume(app, cfg, change("Second revision, no edits"))
    assert app.get_state(cfg).values["used_ids"] == ["T01", "T02", "T03", "T04"]


def repair_inputs(llm):
    return [user for _, system, user in llm.calls if system == REPAIR_SYSTEM]


def test_repair_prompt_lists_retired_ids(start, cfg):
    # Bug: repair never told which ids were removed, so it can re-create "T04" for a new task.
    app, _, llm, _ = start(
        tool_calls(("RemoveTask", {"task_id": "T04"}),
                   ("UpdateTask", {"task_id": "T01", "changes": {"depends_on": ["T02"]}})),
        agent_reply(),
        repair_plans=[cyclic_plan()] * MAX_REPAIR_ATTEMPTS,
    )
    resume(app, cfg, change("Drop T04, T01 after T02"))
    prompts = repair_inputs(llm)
    assert len(prompts) == MAX_REPAIR_ATTEMPTS
    assert prompts[0].endswith("\n\nNever use these ids for new tasks (they were removed): T04")


def test_repair_prompt_has_no_retired_line_when_nothing_was_removed(start, cfg):
    # Bug: the retired-ids line is added unconditionally (with an empty list), confusing the model.
    app, _, llm, _ = start(
        tool_calls(("UpdateTask", {"task_id": "T01", "changes": {"depends_on": ["T02"]}})),
        agent_reply(),
        repair_plans=[cyclic_plan()] * MAX_REPAIR_ATTEMPTS,
    )
    resume(app, cfg, change("T01 after T02"))
    prompts = repair_inputs(llm)
    assert prompts
    assert all("Never use these ids" not in p for p in prompts)


def test_initial_repair_prompt_has_no_retired_line(monkeypatch, cfg):
    # Bug: state["used_ids"] read without .get -> KeyError when the first plan needs repair.
    llm = ScriptedLLM(REQ, plan=cyclic_plan(), repair_plans=[make_multi_module_plan()])
    monkeypatch.setattr(graph_module, "call_structured", llm)
    monkeypatch.setattr(graph_module, "call_agent", ScriptedAgent())
    build_graph(None, checkpointer=make_checkpointer(), human_review=True).invoke({"idea": IDEA}, cfg)
    [prompt] = repair_inputs(llm)
    assert "Never use these ids" not in prompt


def test_step_limit_log_when_last_response_still_has_tool_calls(start, cfg):
    # Bug: hitting the limit is silent, so dropped edits look like a finished revision.
    responses = [tool_calls(("GetTask", {"task_id": "T01"}), ("GetTask", {"task_id": "T02"}))
                 for _ in range(MAX_AGENT_STEPS)]
    app, _, _, _ = start(*responses)
    result = resume(app, cfg, change("Keep going"))
    assert "revise: step limit reached, 2 tool calls not run" in result["log"]


def test_no_step_limit_log_when_agent_finishes_on_last_step(start, cfg):
    # Bug: warning keyed on steps alone, so a clean finish on step MAX is reported as cut off.
    responses = [tool_calls(("GetTask", {"task_id": "T01"})) for _ in range(MAX_AGENT_STEPS - 1)]
    app, agent, _, _ = start(*responses, agent_reply())
    result = resume(app, cfg, change("Finish on the last step"))
    assert len(agent.calls) == MAX_AGENT_STEPS
    assert not any("step limit reached" in line for line in result["log"])


def test_rollback_notice_names_the_change_and_restored_version(start, cfg):
    # Bug: rollback is silent (or uses the discard text), so the user thinks the change never happened.
    app, _, result = run_cycle_revision(start, cfg)
    assert payload(result)["notice"] == (
        "The change 'T01 should wait for T02' produced a plan that could not be repaired, "
        "so it was rolled back to v1."
    )


def test_discard_notice_names_both_versions(start, cfg):
    # Bug: discard reuses the failure text, telling the user their change "could not be repaired".
    _, result = discard_after_one_revision(start, cfg)
    assert payload(result)["notice"] == "Discarded v2. Back to v1."


def test_notice_cleared_on_next_review(start, cfg):
    # Bug: notice never reset -> every later review repeats "Discarded v2".
    app, _, _, _ = start(tool_calls(("RemoveTask", {"task_id": "T04"})), agent_reply(), agent_reply())
    resume(app, cfg, change("Drop T04"))
    resume(app, cfg, DISCARD)
    result = resume(app, cfg, change("Anything"))
    assert payload(result)["notice"] is None

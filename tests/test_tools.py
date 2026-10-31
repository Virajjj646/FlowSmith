"""Step 6b: apply_tool_call, the bridge between model tool calls and plan_ops."""

import json

import pytest
from helpers import make_multi_module_plan, make_plan, make_task, new_task_args
from pydantic import ValidationError

from flowsmith.plan_ops import PlanEditError
from flowsmith.tools import TOOLS, TOOLS_BY_NAME, apply_tool_call


@pytest.fixture
def plan():
    # T01 (Core) <- T02 (Auth) <- T04 (Auth);  T01 <- T03 (Core)
    return make_multi_module_plan()


def ids(plan):
    return [t.id for t in plan.tasks]


def task(plan, task_id):
    return next(t for t in plan.tasks if t.id == task_id)


# ---------------------------------------------------------------------------
# 1. Happy paths
# ---------------------------------------------------------------------------

def test_get_task_returns_task_json(plan):
    # Bug: returns repr()/str() of the Task, which the model can't parse reliably.
    _, output = apply_tool_call(plan, "GetTask", {"task_id": "T02"}, [])
    assert json.loads(output) == task(plan, "T02").model_dump()


def test_get_task_leaves_plan_unchanged(plan):
    # Bug: a read tool returns a rebuilt/modified plan, polluting the revision diff.
    new_plan, _ = apply_tool_call(plan, "GetTask", {"task_id": "T02"}, [])
    assert new_plan == plan


def test_get_dependents_returns_json_list(plan):
    # Bug: Python list repr ("['T02', 'T03']") instead of JSON (double quotes).
    _, output = apply_tool_call(plan, "GetDependents", {"task_id": "T01"}, [])
    assert json.loads(output) == ["T02", "T03"]


def test_get_dependents_leaves_plan_unchanged(plan):
    # Bug: a read tool returns a modified plan.
    new_plan, _ = apply_tool_call(plan, "GetDependents", {"task_id": "T01"}, [])
    assert new_plan == plan


def test_add_task_reports_generated_id(plan):
    # Bug: result text omits the id, so the model can't reference the task in later calls.
    new_plan, output = apply_tool_call(plan, "AddTask", new_task_args(), [])
    assert output == "Added T05."
    assert ids(new_plan) == ["T01", "T02", "T03", "T04", "T05"]


def test_add_task_stores_the_given_fields(plan):
    # Bug: NewTask dumped with wrong keys/aliases, so fields are lost on the way to plan_ops.
    new_plan, _ = apply_tool_call(
        plan, "AddTask", new_task_args(title="Invoices", module="Auth", depends_on=["T02"]), []
    )
    added = task(new_plan, "T05")
    assert (added.title, added.module, added.depends_on) == ("Invoices", "Auth", ["T02"])


def test_update_task_changes_only_provided_fields(plan):
    # Bug: model_dump() without exclude_none -> every unspecified field is set to None and rejected/wiped.
    new_plan, output = apply_tool_call(
        plan, "UpdateTask", {"task_id": "T03", "changes": {"size": "L"}}, []
    )
    assert task(new_plan, "T03").model_dump() == {**task(plan, "T03").model_dump(), "size": "L"}
    assert output == "Updated T03: size."


def test_update_task_result_lists_every_changed_field(plan):
    # Bug: result reports only the first field, so the model can't confirm the full edit.
    _, output = apply_tool_call(
        plan, "UpdateTask", {"task_id": "T03", "changes": {"title": "x", "priority": "could"}}, []
    )
    assert output == "Updated T03: title, priority."


def test_update_task_can_clear_a_list_field(plan):
    # Bug: falsy filter (if v) instead of exclude_none -> depends_on=[] is silently dropped.
    new_plan, _ = apply_tool_call(
        plan, "UpdateTask", {"task_id": "T03", "changes": {"depends_on": []}}, []
    )
    assert task(new_plan, "T03").depends_on == []


def test_remove_task_removes_leaf(plan):
    # Bug: RemoveTask routed to the wrong op, or result text missing the id.
    new_plan, output = apply_tool_call(plan, "RemoveTask", {"task_id": "T04"}, [])
    assert ids(new_plan) == ["T01", "T02", "T03"]
    assert output == "Removed T04."


def test_add_module_adds_module(plan):
    # Bug: module name/description swapped when unpacking the match pattern.
    new_plan, output = apply_tool_call(
        plan, "AddModule", {"name": "Billing", "description": "payments"}, []
    )
    assert new_plan.modules[-1].model_dump() == {"name": "Billing", "description": "payments"}
    assert output == "Added module 'Billing'."


# ---------------------------------------------------------------------------
# 2. Unknown tool
# ---------------------------------------------------------------------------

def test_unknown_tool_lists_available_tools(plan):
    # Bug: KeyError on TOOLS_BY_NAME[name], or the message doesn't help the model pick a real tool.
    with pytest.raises(PlanEditError) as exc:
        apply_tool_call(plan, "DeleteEverything", {}, [])
    message = str(exc.value)
    assert "Unknown tool 'DeleteEverything'" in message
    for name in TOOLS_BY_NAME:
        assert name in message


# ---------------------------------------------------------------------------
# 3. Invalid arguments
# ---------------------------------------------------------------------------

INVALID_ARGS = {
    "GetTask missing task_id": ("GetTask", {}, "task_id"),
    "GetTask task_id wrong type": ("GetTask", {"task_id": 5}, "task_id"),
    "UpdateTask size XL": ("UpdateTask", {"task_id": "T01", "changes": {"size": "XL"}}, "changes.size"),
    "UpdateTask changes not object": ("UpdateTask", {"task_id": "T01", "changes": "size=L"}, "changes"),
    "AddTask fields not nested": ("AddTask", {"title": "x"}, "task"),
    "AddTask nested missing field": (
        "AddTask", {"task": {k: v for k, v in new_task_args()["task"].items() if k != "module"}}, "task.module"
    ),
    "AddTask nested bad priority": ("AddTask", new_task_args(priority="urgent"), "task.priority"),
    "AddModule missing description": ("AddModule", {"name": "Billing"}, "description"),
}


@pytest.mark.parametrize("name, args, location", INVALID_ARGS.values(), ids=INVALID_ARGS.keys())
def test_invalid_args_raise_plan_edit_error_naming_the_field(plan, name, args, location):
    # Bug: pydantic ValidationError escapes the tools node and crashes the graph instead of reaching the model.
    with pytest.raises(Exception) as exc:
        apply_tool_call(plan, name, args, [])
    assert type(exc.value) is PlanEditError
    assert not isinstance(exc.value, ValidationError)
    message = str(exc.value)
    assert message.startswith(f"Invalid arguments for {name}: ")
    assert location in message


def test_plan_ops_errors_pass_through_as_plan_edit_error(plan):
    # Bug: apply_tool_call wraps/hides plan_ops errors, losing the actionable message.
    with pytest.raises(PlanEditError, match="Cannot remove T01: T02, T03 depend on it"):
        apply_tool_call(plan, "RemoveTask", {"task_id": "T01"}, [])


# ---------------------------------------------------------------------------
# 4. Empty UpdateTask
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("changes", [{}, {"title": None}], ids=["empty", "all-none"])
def test_update_task_with_no_changes_raises(plan, changes):
    # Bug: an empty edit "succeeds", so the model believes it changed something.
    with pytest.raises(PlanEditError, match="UpdateTask needs at least one field in 'changes'"):
        apply_tool_call(plan, "UpdateTask", {"task_id": "T01", "changes": changes}, [])


# ---------------------------------------------------------------------------
# 5. reserved_ids
# ---------------------------------------------------------------------------

def test_add_task_respects_reserved_ids():
    # Bug: reserved_ids not forwarded to add_task -> a removed task's id is handed out again.
    plan = make_plan(make_task("T01"), make_task("T02"))
    new_plan, output = apply_tool_call(plan, "AddTask", new_task_args(), ["T01", "T02", "T03"])
    assert output == "Added T04."
    assert ids(new_plan) == ["T01", "T02", "T04"]


# ---------------------------------------------------------------------------
# 6. Tool descriptions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool", TOOLS, ids=[t.__name__ for t in TOOLS])
def test_every_tool_has_a_docstring(tool):
    # Bug: a tool without a docstring is sent to the model with no description, so it is misused.
    assert tool.__doc__ and tool.__doc__.strip()
    assert tool.__doc__ is not tool.__base__.__doc__  # not inherited from BaseModel


def test_tools_by_name_matches_tools():
    # Bug: a tool added to TOOLS but not dispatchable (or vice versa).
    assert list(TOOLS_BY_NAME) == [t.__name__ for t in TOOLS]
    assert list(TOOLS_BY_NAME.values()) == TOOLS

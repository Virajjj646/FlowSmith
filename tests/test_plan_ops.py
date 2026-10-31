"""Step 6a: plan editing operations and plan diff."""

import pytest
from helpers import make_multi_module_plan, make_plan, make_req, make_task, task_fields

from flowsmith.plan_ops import (
    EDITABLE_FIELDS,
    PlanDiff,
    PlanEditError,
    TaskChange,
    add_module,
    add_task,
    diff_plans,
    get_dependents,
    next_task_id,
    remove_task,
    update_task,
)
from flowsmith.schemas import Plan
from flowsmith.validation import validate_plan

REQ = make_req("F1", "F2")


@pytest.fixture
def plan():
    return make_multi_module_plan()


def ids(plan):
    return [t.id for t in plan.tasks]


def task(plan, task_id):
    return next(t for t in plan.tasks if t.id == task_id)


# ---------------------------------------------------------------------------
# 0. Fixture sanity
# ---------------------------------------------------------------------------

def test_fixture_plan_is_valid(plan):
    # Bug in the test setup: a broken fixture would make "no errors after editing" meaningless.
    assert validate_plan(plan, REQ).errors == []


# ---------------------------------------------------------------------------
# 1. Immutability
# ---------------------------------------------------------------------------

def test_add_task_leaves_input_plan_unchanged(plan):
    # Bug: plan.tasks.append(task) mutates the caller's plan in place.
    before = plan.model_dump()
    add_task(plan, task_fields())
    assert plan.model_dump() == before


def test_update_task_leaves_input_plan_unchanged(plan):
    # Bug: setattr on the existing Task object instead of building a new one.
    before = plan.model_dump()
    update_task(plan, "T02", {"title": "Changed", "depends_on": []})
    assert plan.model_dump() == before


def test_remove_task_leaves_input_plan_unchanged(plan):
    # Bug: plan.tasks.remove(task) mutates the caller's plan in place.
    before = plan.model_dump()
    remove_task(plan, "T04")
    assert plan.model_dump() == before


def test_add_module_leaves_input_plan_unchanged(plan):
    # Bug: plan.modules.append(module) mutates the caller's plan in place.
    before = plan.model_dump()
    add_module(plan, "Billing", "payments")
    assert plan.model_dump() == before


# ---------------------------------------------------------------------------
# 2. next_task_id
# ---------------------------------------------------------------------------

def test_next_task_id_empty_plan_is_t01():
    # Bug: max() without default -> ValueError on an empty plan.
    assert next_task_id(make_plan()) == "T01"


def test_next_task_id_uses_max_plus_one_not_count():
    # Bug: len(plan.tasks) + 1 -> T03, colliding with ids after a removal.
    assert next_task_id(make_plan(make_task("T01"), make_task("T05"))) == "T06"


def test_next_task_id_zero_pads_to_two_digits():
    # Bug: f"T{n}" without padding -> "T2", breaking T01/T02 ordering.
    assert next_task_id(make_plan(make_task("T01"))) == "T02"


def test_next_task_id_after_t99_is_t100():
    # Bug: padding by slicing/truncating to 2 digits -> "T00" or "T10".
    assert next_task_id(make_plan(make_task("T99"))) == "T100"


@pytest.mark.parametrize("odd_id", ["X9", "Tabc", "T", "t50"])
def test_next_task_id_ignores_non_t_number_ids(odd_id):
    # Bug: int(t.id[1:]) without the isdigit/startswith guard -> ValueError or a wrong max.
    plan = make_plan(make_task("T03"), make_task(odd_id))
    assert next_task_id(plan) == "T04"


# Non-ASCII ids, written as escapes so the file stays ASCII:
#   "T\u00b2"        T + superscript two (Unicode No: str.isdigit() is True, but int() rejects it)
#   "T\uff15\uff10"  T + fullwidth "50" (Unicode Nd: matched by \d, and int() reads it as 50)
@pytest.mark.parametrize(
    "odd_id",
    ["T\u00b2", "Tabc", "T", "t05", " T05", "T05 ", "T\uff15\uff10"],
    ids=["superscript", "letters", "bare-T", "lowercase", "leading-space", "trailing-space", "fullwidth-50"],
)
def test_next_task_id_ignores_malformed_ids_and_uses_max_of_valid(odd_id):
    # Bug: isdigit()/\d instead of an ASCII [0-9] fullmatch -> crash on superscript, or fullwidth counted as 50.
    plan = make_plan(make_task("T01"), make_task(odd_id), make_task("T07"), make_task("T03"))
    assert next_task_id(plan) == "T08"


def test_next_task_id_includes_reserved_ids_in_max():
    # Bug: reserved_ids accepted but ignored -> ids of removed tasks get handed out again.
    plan = make_plan(make_task("T01"), make_task("T02"))
    assert next_task_id(plan, reserved_ids=["T01", "T02", "T09"]) == "T10"


def test_next_task_id_ignores_malformed_reserved_ids():
    # Bug: reserved ids skip the regex filter -> int() crash on a stray value.
    plan = make_plan(make_task("T01"))
    assert next_task_id(plan, reserved_ids=["Tabc", "X99"]) == "T02"


def test_next_task_id_reserved_ids_accepts_any_iterable():
    # Bug: reserved_ids concatenated as a list (ids + reserved_ids) -> TypeError for sets/generators.
    plan = make_plan(make_task("T01"))
    assert next_task_id(plan, reserved_ids=(f"T0{n}" for n in (4, 6))) == "T07"


# ---------------------------------------------------------------------------
# 2b. Ids are never reused within a revision
# ---------------------------------------------------------------------------

def reuse_base():
    return make_plan(
        make_task("T01"), make_task("T02", depends_on=["T01"]), make_task("T04", depends_on=["T02"])
    )


def test_add_task_with_reserved_ids_does_not_reuse_removed_id():
    # Bug: add_task doesn't pass reserved_ids to next_task_id -> new task becomes T03/T04 again.
    base = reuse_base()
    new_plan, new_id = add_task(remove_task(base, "T04"), task_fields(), reserved_ids=ids(base))
    assert new_id == "T05"


def test_diff_with_reserved_ids_reports_remove_and_add_not_change():
    # Bug: a reused id makes the diff show "T04 changed" instead of a removal plus an addition.
    base = reuse_base()
    new_plan, _ = add_task(remove_task(base, "T04"), task_fields(), reserved_ids=ids(base))
    diff = diff_plans(base, new_plan)
    assert (diff.removed, diff.added, diff.changed) == (["T04"], ["T05"], [])


def test_add_task_without_reserved_ids_uses_max_of_current_plan():
    # Pins the default: without reserved_ids only the current plan counts (T01, T02 -> T03).
    # Bug: default changed silently (e.g. a hidden global counter), so callers can't predict ids.
    base = reuse_base()
    _, new_id = add_task(remove_task(base, "T04"), task_fields())
    assert new_id == "T03"


# ---------------------------------------------------------------------------
# 3. add_task
# ---------------------------------------------------------------------------

def test_add_task_returns_new_plan_and_new_id(plan):
    # Bug: returns only the plan, or the id of the wrong task.
    new_plan, new_id = add_task(plan, task_fields(title="Billing page"))
    assert new_id == "T05"
    assert ids(new_plan) == ["T01", "T02", "T03", "T04", "T05"]
    assert task(new_plan, "T05").title == "Billing page"


def test_add_task_rejects_id_in_fields(plan):
    # Bug: caller-supplied id accepted (or silently overwritten) -> duplicate ids possible.
    with pytest.raises(PlanEditError, match="Unknown fields: id"):
        add_task(plan, task_fields(id="T02"))


def test_add_task_rejects_unknown_fields(plan):
    # Bug: extra keys forwarded to Task and surfacing as a raw pydantic error, or ignored.
    with pytest.raises(PlanEditError, match="Unknown fields: estimate, owner"):
        add_task(plan, task_fields(owner="ana", estimate=3))


def test_add_task_rejects_unknown_module(plan):
    # Bug: module not checked -> task in a module that doesn't exist.
    with pytest.raises(PlanEditError, match="Module 'Billing' does not exist"):
        add_task(plan, task_fields(module="Billing"))


def test_add_task_rejects_unknown_dependency(plan):
    # Bug: depends_on not checked -> dangling reference only caught much later.
    with pytest.raises(PlanEditError, match="unknown tasks: T99"):
        add_task(plan, task_fields(depends_on=["T01", "T99"]))


@pytest.mark.parametrize(
    "field, value, allowed",
    [("size", "XL", "'S', 'M' or 'L'"), ("priority", "urgent", "'must', 'should' or 'could'")],
)
def test_add_task_rejects_invalid_literal_with_readable_message(plan, field, value, allowed):
    # Bug: pydantic ValidationError leaks out instead of a one-line PlanEditError naming the field.
    with pytest.raises(PlanEditError) as exc:
        add_task(plan, task_fields(**{field: value}))
    assert f"{field}:" in str(exc.value)
    assert allowed in str(exc.value)


def test_add_task_missing_required_fields_are_named(plan):
    # Bug: only the first missing field is reported, so the LLM fixes them one call at a time.
    with pytest.raises(PlanEditError) as exc:
        add_task(plan, {"title": "Only a title"})
    message = str(exc.value)
    for field in EDITABLE_FIELDS - {"title"}:
        assert f"{field}: Field required" in message


def test_add_task_can_depend_on_existing_tasks(plan):
    # Bug: references checked against the new task's own plan slice, rejecting valid deps.
    new_plan, new_id = add_task(plan, task_fields(depends_on=["T03", "T04"]))
    assert task(new_plan, new_id).depends_on == ["T03", "T04"]


@pytest.mark.parametrize("bad", [None, ["title", "x"], "title=x"], ids=["None", "list", "str"])
def test_add_task_rejects_non_dict_fields(plan, bad):
    # Bug: no type check -> TypeError/AttributeError escapes instead of a PlanEditError the LLM can read.
    with pytest.raises(PlanEditError, match=f"fields must be an object of task fields, got {type(bad).__name__}"):
        add_task(plan, bad)


# ---------------------------------------------------------------------------
# 4. update_task
# ---------------------------------------------------------------------------

def test_update_task_changes_only_the_given_fields(plan):
    # Bug: changes rebuilt from defaults, wiping fields that weren't mentioned.
    new_plan = update_task(plan, "T02", {"title": "Login", "size": "L"})
    before, after = task(plan, "T02").model_dump(), task(new_plan, "T02").model_dump()
    assert after == {**before, "title": "Login", "size": "L"}


def test_update_task_leaves_other_tasks_identical(plan):
    # Bug: the list comprehension replaces every task, or the wrong one.
    new_plan = update_task(plan, "T02", {"title": "Login"})
    for tid in ("T01", "T03", "T04"):
        assert task(new_plan, tid) == task(plan, tid)


def test_update_task_rejects_id_and_lists_editable_fields(plan):
    # Bug: "id" is editable -> renumbering breaks every depends_on that points at the task.
    with pytest.raises(PlanEditError) as exc:
        update_task(plan, "T02", {"id": "T09"})
    message = str(exc.value)
    assert "Cannot change: id" in message
    assert f"Editable fields: {', '.join(sorted(EDITABLE_FIELDS))}." in message


def test_update_task_rejects_non_editable_field(plan):
    # Bug: unknown keys silently ignored, so the caller thinks the edit happened.
    with pytest.raises(PlanEditError, match="Cannot change: owner"):
        update_task(plan, "T02", {"owner": "ana"})


def test_update_task_unknown_id_lists_existing_ids(plan):
    # Bug: StopIteration/KeyError instead of a message the LLM can recover from.
    with pytest.raises(PlanEditError, match=r"Task T99 does not exist\. Existing ids: T01, T02, T03, T04\."):
        update_task(plan, "T99", {"title": "x"})


def test_update_task_rejects_self_dependency(plan):
    # Bug: self-check missing -> a one-task cycle reaches the scheduler.
    with pytest.raises(PlanEditError, match="T02 cannot depend on itself"):
        update_task(plan, "T02", {"depends_on": ["T01", "T02"]})


def test_update_task_revalidates_values(plan):
    # Bug: model_copy(update=...) skips validation, so size="XL" is stored silently.
    with pytest.raises(PlanEditError, match="size:"):
        update_task(plan, "T02", {"size": "XL"})


def test_update_task_rejects_unknown_module(plan):
    # Bug: references only checked in add_task, not on update.
    with pytest.raises(PlanEditError, match="Module 'Billing' does not exist"):
        update_task(plan, "T02", {"module": "Billing"})


def test_update_task_keeps_position_in_list(plan):
    # Bug: task removed and re-appended, so the edited task jumps to the end.
    new_plan = update_task(plan, "T02", {"title": "Login"})
    assert ids(new_plan) == ["T01", "T02", "T03", "T04"]


def test_update_task_with_empty_changes_gives_equal_plan(plan):
    # Bug: rebuilding the task drops or normalises a field even when nothing changed.
    assert update_task(plan, "T02", {}) == plan


@pytest.mark.parametrize("bad", [None, ["title", "x"], "title=x"], ids=["None", "list", "str"])
def test_update_task_rejects_non_dict_changes(plan, bad):
    # Bug: no type check -> set(None) TypeError, or a string's characters treated as field names.
    with pytest.raises(PlanEditError, match=f"changes must be an object of task fields, got {type(bad).__name__}"):
        update_task(plan, "T02", bad)


# ---------------------------------------------------------------------------
# 5. remove_task
# ---------------------------------------------------------------------------

def test_remove_task_removes_a_leaf(plan):
    # Bug: filter condition inverted, or the wrong task removed.
    assert ids(remove_task(plan, "T04")) == ["T01", "T02", "T03"]


def test_remove_task_refuses_when_dependents_exist_and_names_them_all(plan):
    # Bug: cascading delete, or the message names only the first dependent.
    with pytest.raises(PlanEditError) as exc:
        remove_task(plan, "T01")
    assert str(exc.value).startswith("Cannot remove T01: T02, T03 depend on it.")


def test_remove_task_rejects_unknown_id(plan):
    # Bug: removing a missing id silently returns an identical plan, hiding the LLM's mistake.
    with pytest.raises(PlanEditError, match="Task T99 does not exist"):
        remove_task(plan, "T99")


# ---------------------------------------------------------------------------
# 6. get_dependents
# ---------------------------------------------------------------------------

def test_get_dependents_returns_direct_dependents_only(plan):
    # Bug: transitive walk -> T04 (which depends on T02, not T01) reported as blocking T01.
    assert get_dependents(plan, "T01") == ["T02", "T03"]


def test_get_dependents_of_leaf_is_empty(plan):
    # Bug: substring match on ids ("T0" in "T04") instead of list membership.
    assert get_dependents(plan, "T04") == []


def test_get_dependents_unknown_id_raises(plan):
    # Bug: unknown id returns [] -> remove_task's precondition looks satisfied.
    with pytest.raises(PlanEditError, match="Task T99 does not exist"):
        get_dependents(plan, "T99")


# ---------------------------------------------------------------------------
# 7. add_module
# ---------------------------------------------------------------------------

def test_add_module_appends_module(plan):
    # Bug: module list replaced instead of extended.
    new_plan = add_module(plan, "Billing", "payments")
    assert [m.name for m in new_plan.modules] == ["Core", "Auth", "Billing"]


def test_task_can_use_newly_added_module(plan):
    # Bug: module check reads a cached/stale module set from the original plan.
    new_plan, new_id = add_task(add_module(plan, "Billing", "payments"), task_fields(module="Billing"))
    assert task(new_plan, new_id).module == "Billing"


def test_add_module_rejects_duplicate_name(plan):
    # Bug: duplicate allowed -> two modules with the same name confuse task.module lookups.
    with pytest.raises(PlanEditError, match="Module 'Auth' already exists"):
        add_module(plan, "Auth", "again")


# ---------------------------------------------------------------------------
# 8. Error message quality (every failure case above)
# ---------------------------------------------------------------------------

FAILURES = {
    "add_task: id given": (lambda p: add_task(p, task_fields(id="T02")), "id"),
    "add_task: unknown field": (lambda p: add_task(p, task_fields(owner="ana")), "owner"),
    "add_task: unknown module": (lambda p: add_task(p, task_fields(module="Billing")), "Billing"),
    "add_task: unknown dep": (lambda p: add_task(p, task_fields(depends_on=["T99"])), "T99"),
    "add_task: bad size": (lambda p: add_task(p, task_fields(size="XL")), "size"),
    "add_task: bad priority": (lambda p: add_task(p, task_fields(priority="urgent")), "priority"),
    "add_task: missing fields": (lambda p: add_task(p, {"title": "x"}), "description"),
    "update_task: id": (lambda p: update_task(p, "T02", {"id": "T09"}), "id"),
    "update_task: unknown field": (lambda p: update_task(p, "T02", {"owner": "a"}), "owner"),
    "update_task: unknown task": (lambda p: update_task(p, "T99", {"title": "x"}), "T99"),
    "update_task: self dep": (lambda p: update_task(p, "T02", {"depends_on": ["T02"]}), "T02"),
    "update_task: unknown dep": (lambda p: update_task(p, "T02", {"depends_on": ["T77"]}), "T77"),
    "update_task: bad size": (lambda p: update_task(p, "T02", {"size": "XL"}), "size"),
    "update_task: unknown module": (lambda p: update_task(p, "T02", {"module": "Ops"}), "Ops"),
    "remove_task: has dependents": (lambda p: remove_task(p, "T01"), "T01"),
    "remove_task: unknown task": (lambda p: remove_task(p, "T99"), "T99"),
    "get_dependents: unknown task": (lambda p: get_dependents(p, "T99"), "T99"),
    "add_module: duplicate": (lambda p: add_module(p, "Auth", "again"), "Auth"),
    "add_task: None": (lambda p: add_task(p, None), "fields"),
    "add_task: list": (lambda p: add_task(p, ["title"]), "fields"),
    "add_task: str": (lambda p: add_task(p, "title=x"), "fields"),
    "update_task: None": (lambda p: update_task(p, "T02", None), "changes"),
    "update_task: list": (lambda p: update_task(p, "T02", ["title"]), "changes"),
    "update_task: str": (lambda p: update_task(p, "T02", "title=x"), "changes"),
}


@pytest.mark.parametrize("call, culprit", FAILURES.values(), ids=FAILURES.keys())
def test_error_is_plan_edit_error_on_one_line_naming_the_culprit(plan, call, culprit):
    # Bug: a raw ValidationError/KeyError escapes, or a multi-line pydantic dump reaches the LLM.
    with pytest.raises(Exception) as exc:
        call(plan)
    assert type(exc.value) is PlanEditError
    message = str(exc.value)
    assert "\n" not in message
    assert culprit in message


# ---------------------------------------------------------------------------
# 9. diff_plans
# ---------------------------------------------------------------------------

def test_diff_of_identical_plans_is_empty(plan):
    # Bug: comparing Task objects by identity, so equal copies count as changed.
    diff = diff_plans(plan, plan.model_copy(deep=True))
    assert diff.is_empty
    assert diff.locality == 1.0


def test_diff_detects_added_task(plan):
    # Bug: set difference taken the wrong way round (old - new).
    new_plan, new_id = add_task(plan, task_fields())
    assert diff_plans(plan, new_plan).added == [new_id]


def test_diff_detects_removed_task(plan):
    # Bug: removed tasks counted as unchanged or dropped from the report.
    assert diff_plans(plan, remove_task(plan, "T04")).removed == ["T04"]


def test_diff_reports_exact_changed_field_names(plan):
    # Bug: whole task reported as changed without the fields, or every field listed.
    new_plan = update_task(plan, "T03", {"size": "L", "acceptance_criteria": ["a", "b"]})
    assert diff_plans(plan, new_plan).changed == [
        TaskChange(id="T03", fields=["size", "acceptance_criteria"])
    ]


def test_diff_detects_added_module(plan):
    # Bug: modules ignored by the diff, so new modules are invisible to the reviewer.
    diff = diff_plans(plan, add_module(plan, "Billing", "payments"))
    assert diff.added_modules == ["Billing"]
    assert not diff.is_empty


def test_diff_counts_unchanged_tasks(plan):
    # Bug: unchanged = len(old) instead of shared minus changed.
    new_plan = remove_task(update_task(plan, "T02", {"title": "x"}), "T04")
    assert diff_plans(plan, new_plan).unchanged == 2  # T01, T03


def test_diff_locality_formula(plan):
    # Bug: removed tasks left out of the denominator -> locality overstated after deletions.
    # Add before removing: next_task_id is max+1 of the *current* plan, so adding after
    # removing T04 would reuse "T04" and the diff would see a change, not add + remove.
    new_plan, _ = add_task(update_task(plan, "T02", {"title": "x"}), task_fields())
    new_plan = remove_task(new_plan, "T04")
    diff = diff_plans(plan, new_plan)
    assert (diff.unchanged, len(diff.changed), len(diff.removed)) == (2, 1, 1)
    assert diff.locality == 2 / 4


def test_diff_locality_counts_removed_tasks():
    # Bug: denominator ignores removals -> deleting half the plan still reports locality 1.0.
    diff = PlanDiff(added=[], removed=["T02"], changed=[], unchanged=1, added_modules=[])
    assert diff.locality == 0.5


def test_diff_locality_ignores_added_tasks():
    # Bug: added tasks in the denominator -> pure additions look like the plan was disturbed.
    diff = PlanDiff(added=["T05", "T06"], removed=[], changed=[], unchanged=4, added_modules=[])
    assert diff.locality == 1.0


def test_diff_from_empty_old_plan_has_locality_one():
    # Bug: unconditional division -> ZeroDivisionError when the old plan had no tasks.
    new_plan = make_plan(make_task("T01"))
    diff = diff_plans(make_plan(), new_plan)
    assert diff.added == ["T01"]
    assert diff.locality == 1.0


def test_diff_lists_are_sorted(plan):
    # Bug: built from set iteration order, so the same edit yields differently ordered diffs.
    new_plan, _ = add_task(plan, task_fields())  # adds first, so ids aren't reused
    new_plan, _ = add_task(new_plan, task_fields())
    for tid in ("T04", "T03"):
        new_plan = remove_task(new_plan, tid)
    for tid in ("T02", "T01"):
        new_plan = update_task(new_plan, tid, {"title": f"changed {tid}"})
    diff = diff_plans(plan, new_plan)
    assert diff.removed == ["T03", "T04"]
    assert [c.id for c in diff.changed] == ["T01", "T02"]
    assert diff.added == ["T05", "T06"]


# ---------------------------------------------------------------------------
# 10. Scenario
# ---------------------------------------------------------------------------

def edit_scenario(plan):
    """Add module Billing, add a Billing task after T02, resize T03, remove leaf T04."""
    new_plan = add_module(plan, "Billing", "payments")
    new_plan, new_id = add_task(
        new_plan, task_fields(module="Billing", depends_on=["T02"], covers=["F2"])
    )
    new_plan = update_task(new_plan, "T03", {"size": "L"})
    new_plan = remove_task(new_plan, "T04")
    return new_plan, new_id


def test_scenario_diff_reports_exactly_the_edits(plan):
    # Bug: any op touching tasks it shouldn't (e.g. update rewriting all tasks) shows up here.
    new_plan, new_id = edit_scenario(plan)
    assert diff_plans(plan, new_plan) == PlanDiff(
        added=[new_id],
        removed=["T04"],
        changed=[TaskChange(id="T03", fields=["size"])],
        unchanged=2,
        added_modules=["Billing"],
    )


def test_scenario_result_passes_validation(plan):
    # Bug: an op produces a structurally broken plan (unknown module, dangling dep, lost coverage).
    new_plan, _ = edit_scenario(plan)
    assert validate_plan(new_plan, REQ).errors == []


# ---------------------------------------------------------------------------
# 11. Documented limitation: cycles are the validator's job
# ---------------------------------------------------------------------------

# update_task only checks what it can see locally (ids, module, values, self-dependency).
# A cycle through two or more tasks is a global property, so it is deliberately left to
# validate_plan. These tests pin that split down: if someone adds a cycle check to
# update_task, the first test fails and this comment must be revisited.

def test_update_task_does_not_detect_two_task_cycle(plan):
    # Bug (contract change): update_task starts refusing cycles without the docs/tests being updated.
    update_task(plan, "T01", {"depends_on": ["T02"]})  # T02 already depends on T01


def test_validate_plan_reports_cycle_created_by_update_task(plan):
    # Bug: the validator misses cycles created through edits, so nothing ever catches them.
    new_plan = update_task(plan, "T01", {"depends_on": ["T02"]})
    codes = [e.code for e in validate_plan(new_plan, REQ).errors]
    assert "cycle" in codes

import pytest

from flowsmith.scheduling import schedule_plan, topological_order
from helpers import make_plan, make_task


def position(order):
    return {task_id: i for i, task_id in enumerate(order)}


def test_dependencies_come_first():
    plan = make_plan(
        make_task("T03", depends_on=["T02"]),
        make_task("T02", depends_on=["T01"]),
        make_task("T01"),
    )
    assert schedule_plan(plan).order == ["T01", "T02", "T03"]


def test_must_before_could_when_both_ready():
    plan = make_plan(
        make_task("T01", priority="could"),
        make_task("T02", priority="must"),
    )
    assert schedule_plan(plan).order == ["T02", "T01"]


def test_every_dependency_respected_in_bigger_plan():
    plan = make_plan(
        make_task("T01"),
        make_task("T02", depends_on=["T01"]),
        make_task("T03", depends_on=["T01"], priority="should"),
        make_task("T04", depends_on=["T02", "T03"]),
        make_task("T05", depends_on=["T04"], priority="could"),
    )
    schedule = schedule_plan(plan)
    pos = position(schedule.order)
    for t in plan.tasks:
        for d in t.depends_on:
            assert pos[d] < pos[t.id]


def test_priority_is_raised_for_dependencies():
    plan = make_plan(
        make_task("T01", priority="could"),
        make_task("T02", depends_on=["T01"], priority="must"),
    )
    schedule = schedule_plan(plan)
    assert schedule.effective_priority["T01"] == "must"
    assert [m.name for m in schedule.milestones] == ["MVP"]


def test_milestones_and_waves():
    plan = make_plan(
        make_task("T01"),
        make_task("T02", depends_on=["T01"]),
        make_task("T03", depends_on=["T01"]),
        make_task("T04", depends_on=["T02"], priority="should"),
    )
    mvp, release2 = schedule_plan(plan).milestones
    assert mvp.waves == [["T01"], ["T02", "T03"]]
    assert release2.waves == [["T04"]]


def test_critical_path_follows_the_longest_chain():
    plan = make_plan(
        make_task("T01", size="S"),
        make_task("T02", depends_on=["T01"], size="L"),
        make_task("T03", depends_on=["T01"], size="S"),
        make_task("T04", depends_on=["T02", "T03"], size="M"),
    )
    schedule = schedule_plan(plan)
    assert schedule.mvp_critical_path == ["T01", "T02", "T04"]
    assert schedule.mvp_points == 1 + 4 + 2


def test_critical_path_ignores_later_releases():
    plan = make_plan(
        make_task("T01", size="S"),
        make_task("T02", depends_on=["T01"], size="L", priority="could"),
    )
    schedule = schedule_plan(plan)
    assert schedule.mvp_critical_path == ["T01"]
    assert schedule.mvp_points == 1


def test_cycle_raises():
    plan = make_plan(
        make_task("T01", depends_on=["T02"]),
        make_task("T02", depends_on=["T01"]),
    )
    with pytest.raises(ValueError):
        topological_order(plan, {"T01": "must", "T02": "must"})
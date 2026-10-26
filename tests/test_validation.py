from flowsmith.schemas import Feature, Module, Plan, Requirements, Task
from flowsmith.validation import find_cycles, validate_plan


def make_task(id, depends_on=(), module="Core", priority="must", covers=("F1",)):
    return Task(
        id=id,
        title=f"Task {id}",
        description="test task",
        module=module,
        size="S",
        priority=priority,
        depends_on=list(depends_on),
        covers=list(covers),
        acceptance_criteria=["it works"],
    )


def make_plan(*tasks):
    return Plan(modules=[Module(name="Core", description="core")], tasks=list(tasks))


def make_req(*feature_ids):
    return Requirements(
        app_name="Test",
        summary="test app",
        platform="web",
        target_users=["testers"],
        core_features=[
            Feature(id=fid, name=f"Feature {fid}", description="d", source="stated")
            for fid in (feature_ids or ("F1",))
        ],
        constraints=[],
        assumptions=[],
        gaps=[],
    )


def codes(report):
    return {i.code for i in report.issues}

def test_valid_plan_has_no_issues():
    plan = make_plan(make_task("T01"), make_task("T02", depends_on=["T01"]))
    report = validate_plan(plan, make_req())
    assert report.is_valid
    assert report.issues == []


def test_duplicate_id():
    plan = make_plan(make_task("T01"), make_task("T01"))
    assert "duplicate_id" in codes(validate_plan(plan, make_req()))


def test_unknown_module():
    plan = make_plan(make_task("T01", module="Ghost"))
    assert "unknown_module" in codes(validate_plan(plan, make_req()))


def test_missing_dependency():
    plan = make_plan(make_task("T01", depends_on=["T99"]))
    report = validate_plan(plan, make_req())
    assert "missing_dependency" in codes(report)
    assert "T99" in report.errors[0].message


def test_self_dependency():
    plan = make_plan(make_task("T01", depends_on=["T01"]))
    assert "self_dependency" in codes(validate_plan(plan, make_req()))


def test_cycle_reports_exact_path():
    plan = make_plan(
        make_task("T01", depends_on=["T03"]),
        make_task("T02", depends_on=["T01"]),
        make_task("T03", depends_on=["T02"]),
    )
    cycles = find_cycles(plan)
    assert len(cycles) == 1
    assert set(cycles[0]) == {"T01", "T02", "T03"}
    assert cycles[0][0] == cycles[0][-1]


def test_no_false_cycle_on_diamond():
    plan = make_plan(
        make_task("T01"),
        make_task("T02", depends_on=["T01"]),
        make_task("T03", depends_on=["T01"]),
        make_task("T04", depends_on=["T02", "T03"]),
    )
    assert find_cycles(plan) == []


def test_uncovered_feature():
    plan = make_plan(make_task("T01", covers=["F1"]))
    report = validate_plan(plan, make_req("F1", "F2"))
    assert "uncovered_feature" in codes(report)
    assert not report.is_valid


def test_priority_inversion_is_only_a_warning():
    plan = make_plan(
        make_task("T01", priority="could"),
        make_task("T02", depends_on=["T01"], priority="must"),
    )
    report = validate_plan(plan, make_req())
    assert "priority_inversion" in codes(report)
    assert report.is_valid
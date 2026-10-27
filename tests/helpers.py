from flowsmith.schemas import Feature, Module, Plan, Requirements, Task


def make_task(id, depends_on=(), module="Core", priority="must", covers=("F1",), size="S"):
    return Task(
        id=id,
        title=f"Task {id}",
        description="test task",
        module=module,
        size=size,
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
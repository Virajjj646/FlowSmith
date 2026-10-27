import heapq
from pydantic import BaseModel
from flowsmith.schemas import Plan

PRIORITY_RANK = {"must": 0, "should": 1, "could": 2}
SIZE_POINTS = {"S": 1, "M": 2, "L": 4}
MILESTONE_NAMES = {"must": "MVP", "should": "Release 2", "could": "Release 3"}

class Milestone(BaseModel):
    name: str
    priority: str
    waves: list[list[str]]

class Schedule(BaseModel):
    order: list[str]
    effective_priority: dict[str,str]
    milestones: list[Milestone]
    mvp_critical_path: list[str]
    mvp_points: int

def topological_order(plan: Plan, priority: dict[str,str]) -> list[str]:
    deps = {t.id: set(t.depends_on) for t in plan.tasks}
    dependents: dict[str, list[str]] = {t.id: [] for t in plan.tasks}
    for t in plan.tasks:
        for d in t.depends_on:
            dependents[d].append(t.id)

    remaining = {task_id: len(d) for task_id, d in deps.items()}
    ready = [(PRIORITY_RANK[priority[i]], i) for i, n in remaining.items() if n == 0]
    heapq.heapify(ready)

    order = []
    while ready:
        _,task_id = heapq.heappop(ready)
        order.append(task_id)
        for child in dependents[task_id]:
            remaining[child] -= 1
            if remaining[child] == 0:
                heapq.heappush(ready, (PRIORITY_RANK[priority[child]], child))

    if len(order) != len(plan.tasks):
        raise ValueError("Plan has a cycle; validate it before scheduling.")
    return order

def effective_priorities(plan: Plan) -> dict[str, str]:
    by_id = {t.id: t for t in plan.tasks}
    eff = {t.id: t.priority for t in plan.tasks}
    order = topological_order(plan,eff)
    for task_id in reversed(order):
        for dep in by_id[task_id].depends_on:
            if PRIORITY_RANK[eff[task_id]] < PRIORITY_RANK[eff[dep]]:
                eff[dep] = eff[task_id]
    return eff

def build_milestones(plan: Plan, order: list[str], eff: dict[str,str]) -> list[Milestone]:
    by_id = {t.id: t for t in plan.tasks}
    milstones = []
    for prio in ("must", "should", "could"):
        members = [i for i in order if eff[i] == prio]
        if not members:
            continue
        member_set = set(members)
        level: dict[str, int] = {}
        for task_id in members:
            same_milestones_deps = [d for d in by_id[task_id].depends_on if d in member_set]
            level[task_id] = 1 + max((level[d] for d in same_milestones_deps), default=-1)

        waves = [[] for _ in range(max(level.values()) + 1)]
        for task_id in members:
            waves[level[task_id]].append(task_id)
        milstones.append(Milestone(name=MILESTONE_NAMES[prio], priority=prio, waves=waves))
    return milstones

def critical_path(plan: Plan, order: list[str]) -> tuple[list[str], int]:
    by_id = {t.id: t for t in plan.tasks}
    finish: dict[str, int] = {}
    prev: dict[str, str | None] = {}
    for task_id in order:
        task = by_id[task_id]
        best = max(task.depends_on, key=lambda d: finish[d], default=None)
        start = finish[best] if best else 0
        finish[task_id] = start + SIZE_POINTS[task.size]
        prev[task_id] = best

    end = max(order, key=lambda i: finish[i])
    path = []
    node: str | None = end
    while node:
        path.append(node)
        node = prev[node]
    return list(reversed(path)),finish[end]

def schedule_plan(plan: Plan) -> Schedule:
    eff = effective_priorities(plan)
    order = topological_order(plan, eff)
    mvp_order = [i for i in order if eff[i] == "must"]
    path, points = critical_path(plan, mvp_order)
    return Schedule(
        order=order,
        effective_priority=eff,
        milestones=build_milestones(plan, order, eff),
        mvp_critical_path=path,
        mvp_points=points,
    )


def print_schedule(plan: Plan, schedule: Schedule) -> None:
    by_id = {t.id: t for t in plan.tasks}
    for m in schedule.milestones:
        print(f"\n{m.name} ({m.priority})")
        for n, wave in enumerate(m.waves, start=1):
            labels = []
            for task_id in wave:
                t = by_id[task_id]
                bumped = "*" if t.priority != schedule.effective_priority[task_id] else ""
                labels.append(f"{task_id}{bumped} {t.title} [{t.size}]")
            print(f"  wave {n}: " + " | ".join(labels))
    print(f"\nMVP critical path: {' -> '.join(schedule.mvp_critical_path)} "
          f"({schedule.mvp_points} points)")
    print("* = priority raised because a higher-priority task depends on it")
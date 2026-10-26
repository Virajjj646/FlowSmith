from collections import Counter
from typing import Literal
from pydantic import BaseModel
from flowsmith.schemas import Plan, Requirements

class Issue(BaseModel):
    code: str
    level: Literal["error", "warning"]
    message: str
    tasks_ids: list[str] = []

class ValidationReport(BaseModel):
    issues: list[Issue]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors

def check_duplicate_ids(plan: Plan) -> list[Issue]:
    counts = Counter(t.id for t in plan.tasks)
    return[
        Issue(
            code="duplicate_id",
            level="error",
            message=f"Task id {task_id} is used {n} times. Every task id must be unique",
            tasks_ids=[task_id],
        )
        for task_id, n in counts.items()
        if n>1
    ]

def check_modules(plan:Plan) -> list[Issue]:
    module_names = {m.name for m in plan.modules}
    used = {t.module for t in plan.tasks}
    issues = [
        Issue(
            code="unknown_module",
            level="error",
            message=f"{t.id} belongs to module '{t.module}', which does not exist. "
            f"Valid modules: {sorted(module_names)}.",
            tasks_ids=[t.id],
        )
        for t in plan.tasks
        if t.module not in module_names
    ]
    issues += [
        Issue(
            code="empty_module",
            level="warning",
            message=f"Module '{name}' has no tasks",
        )
        for name in sorted(module_names - used)
    ]
    return issues

def check_dependencies(plan: Plan) -> list[Issue]:
    ids = {t.id for t in plan.tasks}
    issues =[]
    for t in plan.tasks:
        for dep in t.depends_on:
            if dep == t.id:
                issues.append(Issue(
                    code="self_dependency",
                    level="error",
                    message=f"{t.id} depends on itself.",
                    tasks_ids=[t.id],
                ))
            elif dep not in ids:
                issues.append(Issue(
                    code="missing_dependency",
                    level="error",
                    message=f"{t.id} depends on {dep}, which does not exist.",
                    tasks_ids=[t.id],
                ))
    return issues

WHITE, GRAY, BLACK = 0, 1, 2
def find_cycles(plan:Plan) -> list[list[str]]:
    ids = {t.id for t in plan.tasks}
    graph = {
        t.id: [d for d in t.depends_on if d in ids and  d != t.id]
        for t in plan.tasks
    }
    color = {task_id: WHITE for task_id in graph}
    path: list[str] = []
    cycles: list[list[str]] = []
    seen: set[frozenset[str]] = set()

    def visit(node: str) -> None:
        color[node] = GRAY
        path.append(node)
        for dep in graph[node]:
            if color[dep] == GRAY:
                cycle = path[path.index(dep): ] + [dep]
                key = frozenset(cycle)
                if key not in seen:
                    seen.add(key)
                    cycles.append(cycle)
            elif color[dep] == WHITE:
                visit(dep)
        path.pop()
        color[node] = BLACK

    for task_id in graph:
        if color[task_id] == WHITE:
            visit(task_id)
    return cycles

def check_cycles(plan: Plan) -> list[Issue]:
    return [
        Issue(
            code="cycle",
            level="error",
            message=f"Circular dependency: {' -> '.join(cycle)}. "
            "Remove one of these dependencies.",
            tasks_ids=cycle[:-1],
        )
        for cycle in find_cycles(plan)
    ]

def check_coverage(plan: Plan, req: Requirements) -> list[Issue]:
    feature_ids = {f.id for f in req.core_features}
    covered = {fid for t in plan.tasks for fid in t.covers}
    issues= [
        Issue(
            code="uncovered_feature",
            level="error",
            message=f"Feature {f.id} '{f.name}' is not covered by any task."
        )
        for f in req.core_features
        if f.id not in covered
    ]
    issues += [
        Issue(
            code="unknown_feature",
            level="warning",
            message=f"{t.id} covers {fid}, which is not a known feature.",
            tasks_ids=[t.id]
        )
        for t in plan.tasks
        for fid in t.covers
        if fid not in feature_ids
    ]
    return issues

def check_quality(plan: Plan) -> list[Issue]:
    by_id = {t.id: t for t in plan.tasks}
    issues = []
    for t in plan.tasks:
        if not t.acceptance_criteria:
            issues.append(Issue(
                code="no_acceptance_criteria",
                level="warning",
                message=f"{t.id} has no acceptance criteria.",
                tasks_ids=[t.id]
            ))
        if t.priority == "must":
            for dep in t.depends_on:
                if dep in by_id and by_id[dep].priority == "could":
                    issues.append(Issue(
                        code="priority_inversion",
                        level="warning",
                        message=f"{t.id} is 'must' but depends on {dep}, which is only 'could'.",
                        tasks_ids=[t.id,dep],
                    ))
    return issues

PLAN_CHECKS = [check_duplicate_ids, check_modules, check_dependencies, check_cycles, check_quality]


def validate_plan(plan: Plan, req: Requirements) -> ValidationReport:
    issues: list[Issue] = []
    for check in PLAN_CHECKS:
        issues += check(plan)
    issues += check_coverage(plan, req)
    return ValidationReport(issues=issues)

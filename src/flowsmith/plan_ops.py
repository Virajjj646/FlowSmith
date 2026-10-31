import re
from collections.abc import Iterable

from pydantic import BaseModel, ValidationError

from flowsmith.schemas import Module, Plan, Task

EDITABLE_FIELDS = {
    "title", "description", "module", "size", "priority",
    "depends_on", "covers", "acceptance_criteria",
}


class PlanEditError(Exception):
    """An edit could not be applied. The message is written for the LLM to read and fix."""


def get_task(plan: Plan, task_id: str) -> Task:
    for task in plan.tasks:
        if task.id == task_id:
            return task
    existing = ", ".join(t.id for t in plan.tasks)
    raise PlanEditError(f"Task {task_id} does not exist. Existing ids: {existing}.")


def get_dependents(plan: Plan, task_id: str) -> list[str]:
    get_task(plan, task_id)
    return [t.id for t in plan.tasks if task_id in t.depends_on]


TASK_ID = re.compile(r"T([0-9]+)")


def next_task_id(plan: Plan, reserved_ids: Iterable[str] = ()) -> str:
    ids = [t.id for t in plan.tasks] + list(reserved_ids)
    numbers = [int(m.group(1)) for i in ids if (m := TASK_ID.fullmatch(i))]
    return f"T{max(numbers, default=0) + 1:02d}"


def _require_dict(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise PlanEditError(f"{name} must be an object of task fields, got {type(value).__name__}.")
    return value


def _check_references(plan: Plan, task: Task) -> None:
    modules = {m.name for m in plan.modules}
    if task.module not in modules:
        raise PlanEditError(
            f"Module '{task.module}' does not exist. Use one of {sorted(modules)}, "
            "or add the module first."
        )
    ids = {t.id for t in plan.tasks}
    missing = [d for d in task.depends_on if d not in ids]
    if missing:
        raise PlanEditError(f"depends_on refers to unknown tasks: {', '.join(missing)}.")
    if task.id in task.depends_on:
        raise PlanEditError(f"{task.id} cannot depend on itself.")


def _build_task(data: dict) -> Task:
    try:
        return Task.model_validate(data)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        raise PlanEditError(f"Invalid task fields: {problems}") from e


def add_module(plan: Plan, name: str, description: str) -> Plan:
    if any(m.name == name for m in plan.modules):
        raise PlanEditError(f"Module '{name}' already exists.")
    return plan.model_copy(update={"modules": [*plan.modules, Module(name=name, description=description)]})


def add_task(plan: Plan, fields: dict, reserved_ids: Iterable[str] = ()) -> tuple[Plan, str]:
    fields = _require_dict(fields, "fields")
    unknown = set(fields) - EDITABLE_FIELDS
    if unknown:
        raise PlanEditError(f"Unknown fields: {', '.join(sorted(unknown))}.")
    new_id = next_task_id(plan, reserved_ids)
    task = _build_task({**fields, "id": new_id})
    _check_references(plan, task)
    return plan.model_copy(update={"tasks": [*plan.tasks, task]}), new_id


def update_task(plan: Plan, task_id: str, changes: dict) -> Plan:
    changes = _require_dict(changes, "changes")
    unknown = set(changes) - EDITABLE_FIELDS
    if unknown:
        raise PlanEditError(
            f"Cannot change: {', '.join(sorted(unknown))}. "
            f"Editable fields: {', '.join(sorted(EDITABLE_FIELDS))}."
        )
    old = get_task(plan, task_id)
    new = _build_task({**old.model_dump(), **changes})
    _check_references(plan, new)
    tasks = [new if t.id == task_id else t for t in plan.tasks]
    return plan.model_copy(update={"tasks": tasks})


# Every operation returns a new Plan, but the copies are shallow: the new and old plans
# share the Task objects they didn't change. Edit plans only through these functions.
# Never mutate a Task in place (e.g. new_plan.tasks[0].depends_on.append(...)): that
# silently changes the old plan too, and with it any diff or checkpoint that refers to it.
def remove_task(plan: Plan, task_id: str) -> Plan:
    get_task(plan, task_id)
    dependents = get_dependents(plan, task_id)
    if dependents:
        raise PlanEditError(
            f"Cannot remove {task_id}: {', '.join(dependents)} depend on it. "
            "Update or remove those tasks first."
        )
    return plan.model_copy(update={"tasks": [t for t in plan.tasks if t.id != task_id]})


class TaskChange(BaseModel):
    id: str
    fields: list[str]


class PlanDiff(BaseModel):
    added: list[str]
    removed: list[str]
    changed: list[TaskChange]
    unchanged: int
    added_modules: list[str]

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed or self.added_modules)

    @property
    def locality(self) -> float:
        """Share of the original tasks that the revision left untouched."""
        original = self.unchanged + len(self.changed) + len(self.removed)
        return self.unchanged / original if original else 1.0


def diff_plans(old: Plan, new: Plan) -> PlanDiff:
    old_tasks = {t.id: t for t in old.tasks}
    new_tasks = {t.id: t for t in new.tasks}

    changed = []
    for task_id in old_tasks.keys() & new_tasks.keys():
        before, after = old_tasks[task_id].model_dump(), new_tasks[task_id].model_dump()
        fields = [f for f in before if before[f] != after[f]]
        if fields:
            changed.append(TaskChange(id=task_id, fields=fields))

    old_modules = {m.name for m in old.modules}
    return PlanDiff(
        added=sorted(new_tasks.keys() - old_tasks.keys()),
        removed=sorted(old_tasks.keys() - new_tasks.keys()),
        changed=sorted(changed, key=lambda c: c.id),
        unchanged=len(old_tasks.keys() & new_tasks.keys()) - len(changed),
        added_modules=[m.name for m in new.modules if m.name not in old_modules],
    )

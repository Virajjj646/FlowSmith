import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from flowsmith.plan_ops import (
    PlanEditError, add_module, add_task, get_dependents, get_task, remove_task, update_task,
)
from flowsmith.schemas import Plan


class GetTask(BaseModel):
    """Read every field of one task."""
    task_id: str


class GetDependents(BaseModel):
    """List the ids of tasks that directly depend on a task. Check this before removing a task."""
    task_id: str


class AddModule(BaseModel):
    """Add a new module (feature area) that tasks can belong to."""
    name: str
    description: str


class NewTask(BaseModel):
    title: str
    description: str
    module: str = Field(description="Exact name of an existing module")
    size: Literal["S", "M", "L"] = Field(description="S: under a day, M: 1-3 days, L: 3-5 days")
    priority: Literal["must", "should", "could"]
    depends_on: list[str] = Field(description="Ids of existing tasks that must be finished first")
    covers: list[str] = Field(description="Feature ids this task implements, or an empty list")
    acceptance_criteria: list[str]


class AddTask(BaseModel):
    """Add a new task. Its id is generated for you and returned in the result."""
    task: NewTask


class TaskChanges(BaseModel):
    title: str | None = None
    description: str | None = None
    module: str | None = None
    size: Literal["S", "M", "L"] | None = None
    priority: Literal["must", "should", "could"] | None = None
    depends_on: list[str] | None = None
    covers: list[str] | None = None
    acceptance_criteria: list[str] | None = None


class UpdateTask(BaseModel):
    """Change fields of an existing task. Include only the fields you want to change."""
    task_id: str
    changes: TaskChanges


class RemoveTask(BaseModel):
    """Remove a task. Fails if other tasks depend on it."""
    task_id: str


TOOLS = [GetTask, GetDependents, AddModule, AddTask, UpdateTask, RemoveTask]
TOOLS_BY_NAME = {tool.__name__: tool for tool in TOOLS}


def apply_tool_call(plan: Plan, name: str, args: dict, reserved_ids: list[str]) -> tuple[Plan, str]:
    """Run one tool call. Returns the (possibly new) plan and the text the model will see."""
    schema = TOOLS_BY_NAME.get(name)
    if schema is None:
        raise PlanEditError(f"Unknown tool '{name}'. Available: {', '.join(TOOLS_BY_NAME)}.")
    try:
        call = schema.model_validate(args)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        raise PlanEditError(f"Invalid arguments for {name}: {problems}") from e

    match call:
        case GetTask(task_id=task_id):
            return plan, get_task(plan, task_id).model_dump_json()
        case GetDependents(task_id=task_id):
            return plan, json.dumps(get_dependents(plan, task_id))
        case AddModule(name=module_name, description=description):
            return add_module(plan, module_name, description), f"Added module '{module_name}'."
        case AddTask(task=task):
            new_plan, new_id = add_task(plan, task.model_dump(), reserved_ids)
            return new_plan, f"Added {new_id}."
        case UpdateTask(task_id=task_id, changes=changes):
            fields = changes.model_dump(exclude_none=True)
            if not fields:
                raise PlanEditError("UpdateTask needs at least one field in 'changes'.")
            return update_task(plan, task_id, fields), f"Updated {task_id}: {', '.join(fields)}."
        case RemoveTask(task_id=task_id):
            return remove_task(plan, task_id), f"Removed {task_id}."

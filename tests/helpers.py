import itertools

from langchain_core.messages import AIMessage

from flowsmith.prompts import REPAIR_SYSTEM
from flowsmith.schemas import Feature, Gap, Module, Plan, Requirements, Task


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


def make_valid_plan():
    """A plan that passes validation for make_req() (covers F1, no cycles)."""
    return make_plan(make_task("T01"), make_task("T02", depends_on=["T01"]))


def make_multi_module_plan():
    """Valid two-module plan for make_req("F1", "F2").

        T01 (Core)  <- T02 (Auth)  <- T04 (Auth)   leaf
                    <- T03 (Core)                  leaf
    """
    return Plan(
        modules=[
            Module(name="Core", description="core"),
            Module(name="Auth", description="auth"),
        ],
        tasks=[
            make_task("T01", covers=("F1",)),
            make_task("T02", depends_on=["T01"], module="Auth", covers=("F2",)),
            make_task("T03", depends_on=["T01"], covers=("F1",)),
            make_task("T04", depends_on=["T02"], module="Auth", covers=("F2",)),
        ],
    )


def task_fields(**overrides):
    """A complete, valid fields dict for add_task (no id: ids are generated)."""
    fields = dict(
        title="New task",
        description="new",
        module="Core",
        size="M",
        priority="should",
        depends_on=[],
        covers=["F1"],
        acceptance_criteria=["done"],
    )
    return {**fields, **overrides}


def make_gap(question, severity="blocking"):
    return Gap(question=question, severity=severity, reason=f"{severity} reason")


def make_req(*feature_ids, gaps=()):
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
        gaps=list(gaps),
    )


def blocking_req(*questions):
    """Requirements whose gaps are all blocking, one per question."""
    return make_req(gaps=[make_gap(q) for q in questions])


class ScriptedLLM:
    """Stands in for flowsmith.graph.call_structured.

    Requirements calls pop from a scripted queue (so each analyze round can return
    different gaps); every Plan call returns the same valid plan, except repair calls,
    which pop from `repair_plans` when one is given. All calls are recorded as
    (schema, system, user).
    """

    def __init__(self, *requirements, plan=None, repair_plans=None):
        self.requirements = list(requirements)
        self.plan = plan or make_valid_plan()
        self.repair_plans = None if repair_plans is None else list(repair_plans)
        self.calls = []

    def __call__(self, llm, schema, system, user):
        self.calls.append((schema, system, user))
        if schema is Requirements:
            if not self.requirements:
                raise AssertionError("ScriptedLLM ran out of Requirements (unexpected analyze)")
            return self.requirements.pop(0)
        if self.repair_plans is not None and system == REPAIR_SYSTEM:
            if not self.repair_plans:
                raise AssertionError("ScriptedLLM ran out of repair plans (unexpected repair)")
            return self.repair_plans.pop(0)
        return self.plan

    @property
    def analyze_inputs(self):
        return [user for schema, _, user in self.calls if schema is Requirements]

    @property
    def plan_calls(self):
        return [c for c in self.calls if c[0] is Plan]


# ---------------------------------------------------------------------------
# Revision agent fakes (Step 6b)
# ---------------------------------------------------------------------------

_call_ids = itertools.count(1)


def tool_calls(*calls):
    """An AIMessage requesting tools. Each call is (name, args); ids are unique per process.

    Always build a fresh message per scripted step: add_messages stores the message's id,
    so re-using one object would replace the earlier copy instead of appending.
    """
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call_{next(_call_ids)}"} for name, args in calls],
    )


def agent_reply(text="Done."):
    """A final AIMessage with no tool calls: ends the agent loop."""
    return AIMessage(content=text)


def new_task_args(**overrides):
    """Valid AddTask args: {"task": {...}}."""
    return {"task": task_fields(**overrides)}


class ScriptedAgent:
    """Stands in for flowsmith.graph.call_agent. Returns scripted AIMessages in order.

    Every call is recorded as (system, messages, tools); messages is a snapshot copy.
    """

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, llm, system, messages, tools):
        self.calls.append((system, list(messages), tools))
        if not self.responses:
            raise AssertionError("ScriptedAgent ran out of responses (unexpected revise step)")
        return self.responses.pop(0)

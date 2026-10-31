import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.types import interrupt

from flowsmith.llm import call_agent, call_structured
from flowsmith.plan_ops import PlanEditError, diff_plans
from flowsmith.prompts import DECOMPOSE_SYSTEM, REQUIREMENTS_SYSTEM, REPAIR_SYSTEM, REVISE_SYSTEM
from flowsmith.schemas import Clarification, Plan, Requirements
from flowsmith.scheduling import Schedule, schedule_plan
from flowsmith.tools import TOOLS, apply_tool_call
from flowsmith.validation import ValidationReport, validate_plan

MAX_REPAIR_ATTEMPTS = 2
MAX_CLARIFY_ROUNDS = 2
NO_PREFERENCE = "No preference. Choose a sensible default."
MAX_AGENT_STEPS = 8

class AgentState(TypedDict, total=False):
    idea: str
    clarifications: Annotated[list[Clarification], operator.add]
    clarification_rounds: int
    requirements: Requirements
    plan: Plan
    validation: ValidationReport
    schedule: Schedule
    repair_attempts: int
    plan_version: int
    base_plan: Plan | None
    change_request: str
    review_action: str
    messages: Annotated[list[AnyMessage], add_messages]
    agent_steps: int
    used_ids: list[str]
    notice: str | None
    log: Annotated[list[str], operator.add]

def blocking_questions(req: Requirements) -> list[str]:
    return [g.question for g in req.gaps if g.severity == "blocking"]


def build_analyze_input(state: AgentState) -> str:
    text = state["idea"]
    answered = state.get("clarifications", [])
    if answered:
        qa = "\n".join(f"Q: {c.question}\nA: {c.answer}" for c in answered)
        text += f"\n\nAnswers to your earlier questions:\n{qa}"
    return text


def valid_answers(answers: object, questions: list[str]) -> bool:
    return (
        isinstance(answers, list)
        and len(answers) == len(questions)
        and all(isinstance(a, str) for a in answers)
    )


def route_after_analyze(state: AgentState) -> str:
    if (
        blocking_questions(state["requirements"])
        and state.get("clarification_rounds", 0) < MAX_CLARIFY_ROUNDS
    ):
        return "clarify"
    return "decompose"


def plan_overview(plan: Plan) -> str:
    lines = [f"Modules: {', '.join(m.name for m in plan.modules)}"]
    for t in plan.tasks:
        deps = ",".join(t.depends_on) or "-"
        covers = ",".join(t.covers) or "-"
        lines.append(f"{t.id} | {t.module} | {t.size} | {t.priority} | after: {deps} | covers: {covers} | {t.title}")
    return "\n".join(lines)


def build_revision_message(state: AgentState, request: str) -> str:
    return (
        f"Change request: {request}\n\n"
        f"Requirements:\n{state['requirements'].model_dump_json(indent=2)}\n\n"
        f"Current plan (id | module | size | priority | after | covers | title):\n"
        f"{plan_overview(state['plan'])}"
    )


def valid_decision(decision: object, can_discard: bool) -> bool:
    if not isinstance(decision, dict):
        return False
    action = decision.get("action")
    if action == "approve":
        return True
    if action == "discard":
        return can_discard
    if action == "change":
        request = decision.get("request")
        return isinstance(request, str) and bool(request.strip())
    return False


def route_after_review(state: AgentState) -> str:
    return {"approve": "approve", "change": "revise", "discard": "rollback"}[state["review_action"]]


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None) and state.get("agent_steps", 0) < MAX_AGENT_STEPS:
        return "tools"
    return "validate"


def route_after_validate(state: AgentState) -> str:
    if state["validation"].is_valid:
        return "schedule"
    if state.get("repair_attempts", 0) < MAX_REPAIR_ATTEMPTS:
        return "repair"
    if state.get("base_plan") is not None:
        return "rollback"
    return "give_up"

def build_graph(
    llm: ChatGroq,
    checkpointer: BaseCheckpointSaver | None = None,
    human_review: bool = False,
):
    def analyze(state: AgentState) -> dict:
        req = call_structured(llm, Requirements, REQUIREMENTS_SYSTEM, build_analyze_input(state))
        return{
            "requirements": req,
            "log": [f"analyze: {len(req.core_features)} features, {len(req.gaps)} gaps"]
        }

    def clarify(state: AgentState) -> dict:
        questions = blocking_questions(state["requirements"])
        answers = interrupt({"questions": questions})

        while not valid_answers(answers, questions):
            answers = interrupt({
                "questions": questions,
                "error": f"Expected a list of {len(questions)} text answers, got {answers!r}.",
            })

        clarifications = [
            Clarification(question=q, answer=a.strip() or NO_PREFERENCE)
            for q, a in zip(questions, answers)
        ]
        rounds = state.get("clarification_rounds", 0) + 1
        return {
            "clarifications": clarifications,
            "clarification_rounds": rounds,
            "log": [f"clarify: round {rounds}, {len(clarifications)} answers received"],
        }

    def decompose(state: AgentState) -> dict:
        req_json = state["requirements"].model_dump_json(indent=2)
        plan = call_structured(llm,Plan, DECOMPOSE_SYSTEM, f"Requirements:\n{req_json}")
        log = [f"decompose: {len(plan.modules)} modules, {len(plan.tasks)} tasks"]
        unanswered = blocking_questions(state["requirements"])
        if unanswered:
            log.append(f"decompose: planning with {len(unanswered)} unanswered blocking questions")
        return {"plan": plan, "log": log}

    def validate(state: AgentState) -> dict:
        report = validate_plan(state["plan"], state["requirements"])
        return {
            "validation": report,
            "log": [f"validate: {len(report.errors)} errors, {len(report.warnings)} warnings"],
        }

    def repair(state: AgentState) -> dict:
        attempt = state.get("repair_attempts", 0) + 1
        errors = state["validation"].errors
        error_lines = "\n".join(f"- {e.message}" for e in errors)
        user = (
            f"Requirements:\n{state['requirements'].model_dump_json(indent=2)}\n\n"
            f"Current plan:\n{state['plan'].model_dump_json(indent=2)}\n\n"
            f"Errors to fix:\n{error_lines}"
        )
        retired = sorted(set(state.get("used_ids", [])) - {t.id for t in state["plan"].tasks})
        if retired:
            user += f"\n\nNever use these ids for new tasks (they were removed): {', '.join(retired)}"
        plan = call_structured(llm, Plan, REPAIR_SYSTEM, user)
        return {
            "plan": plan,
            "repair_attempts": attempt,
            "log": [f"repair: attempt {attempt}/{MAX_REPAIR_ATTEMPTS}, fixing {len(errors)} errors"],
        }

    def schedule(state: AgentState) -> dict:
        result = schedule_plan(state["plan"])
        return {
            "schedule": result,
            "log": [f"schedule: {len(result.milestones)} milestones"],
        }

    def review(state: AgentState) -> dict:
        base = state.get("base_plan")
        payload = {
            "kind": "review",
            "plan_version": state.get("plan_version", 1),
            "diff": diff_plans(base, state["plan"]).model_dump() if base else None,
            "can_discard": base is not None,
            "notice": state.get("notice"),
        }
        decision = interrupt(payload)
        while not valid_decision(decision, payload["can_discard"]):
            decision = interrupt({
                **payload,
                "error": "Expected {'action': 'approve'}, {'action': 'change', 'request': '...'}"
                + (" or {'action': 'discard'}." if payload["can_discard"] else "."),
            })
        action = decision["action"]
        if action == "approve":
            return {"review_action": "approve", "notice": None,
                    "log": [f"review: approved v{payload['plan_version']}"]}
        if action == "discard":
            return {"review_action": "discard", "notice": None,
                    "log": [f"review: discarded v{payload['plan_version']}"]}
        request = decision["request"].strip()
        return {
            "review_action": "change",
            "change_request": request,
            "base_plan": state["plan"],
            "plan_version": payload["plan_version"] + 1,
            "repair_attempts": 0,
            "agent_steps": 0,
            "used_ids": [t.id for t in state["plan"].tasks],
            "notice": None,
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES),
                         HumanMessage(content=build_revision_message(state, request))],
            "log": [f"review: change requested -> v{payload['plan_version'] + 1}"],
        }

    def rollback(state: AgentState) -> dict:
        version = state.get("plan_version", 1) - 1
        if state.get("review_action") == "discard":
            notice = f"Discarded v{version + 1}. Back to v{version}."
        else:
            notice = (f"The change '{state['change_request']}' produced a plan that could not be "
                      f"repaired, so it was rolled back to v{version}.")
        return {"plan": state["base_plan"], "base_plan": None, "plan_version": version,
                "repair_attempts": 0, "notice": notice, "log": [f"rollback: restored v{version}"]}

    def revise(state: AgentState) -> dict:
        response = call_agent(llm, REVISE_SYSTEM, state["messages"], TOOLS)
        steps = state.get("agent_steps", 0) + 1
        calls = ", ".join(c["name"] for c in response.tool_calls) or "done"
        log = [f"revise: step {steps}/{MAX_AGENT_STEPS}: {calls}"]
        if response.tool_calls and steps >= MAX_AGENT_STEPS:
            log.append(f"revise: step limit reached, {len(response.tool_calls)} tool calls not run")
        return {"messages": [response], "agent_steps": steps, "log": log}

    def tools(state: AgentState) -> dict:
        plan = state["plan"]
        used = set(state["used_ids"])
        results, outcomes = [], []
        for call in state["messages"][-1].tool_calls:
            try:
                plan, output = apply_tool_call(plan, call["name"], call["args"], sorted(used))
                used |= {t.id for t in plan.tasks}
                outcomes.append(f"{call['name']} ok")
            except PlanEditError as e:
                output = f"ERROR: {e}"
                outcomes.append(f"{call['name']} error")
            results.append(ToolMessage(content=output, tool_call_id=call["id"]))
        return {"plan": plan, "messages": results, "used_ids": sorted(used),
                "log": [f"tools: {', '.join(outcomes)}"]}

    builder = StateGraph(AgentState)
    builder.add_node("analyze", analyze)
    builder.add_node("clarify", clarify)
    builder.add_node("decompose", decompose)
    builder.add_node("validate", validate)
    builder.add_node("repair", repair)
    builder.add_node("schedule", schedule)

    builder.add_edge(START, "analyze")
    builder.add_conditional_edges(
        "analyze",
        route_after_analyze,
        {"clarify": "clarify", "decompose": "decompose"},
    )
    builder.add_edge("clarify", "analyze")
    builder.add_edge("decompose", "validate")
    builder.add_edge("repair", "validate")

    validate_routes = {"schedule": "schedule", "repair": "repair", "give_up": END}
    if not human_review:
        builder.add_conditional_edges("validate", route_after_validate, validate_routes)
        builder.add_edge("schedule", END)
        return builder.compile(checkpointer=checkpointer)

    for name, node in [("review", review), ("revise", revise), ("tools", tools), ("rollback", rollback)]:
        builder.add_node(name, node)
    builder.add_conditional_edges("validate", route_after_validate, {**validate_routes, "rollback": "rollback"})
    builder.add_edge("schedule", "review")
    builder.add_conditional_edges("review", route_after_review,
                                  {"approve": END, "revise": "revise", "rollback": "rollback"})
    builder.add_edge("rollback", "validate")
    builder.add_conditional_edges("revise", route_after_agent, {"tools": "tools", "validate": "validate"})
    builder.add_edge("tools", "revise")
    return builder.compile(checkpointer=checkpointer)
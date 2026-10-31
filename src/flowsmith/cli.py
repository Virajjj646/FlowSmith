import sys
import time
import uuid

import groq
from langgraph.types import Command

from flowsmith.analyze_raw import print_requirements
from flowsmith.checkpoint import make_checkpointer
from flowsmith.config import get_settings
from flowsmith.graph import build_graph, MAX_REPAIR_ATTEMPTS
from flowsmith.llm import LLMOutputError, make_llm
from flowsmith.scheduling import print_schedule
from flowsmith.schemas import Plan
from flowsmith.validation import ValidationReport


def print_plan(plan: Plan) -> None:
    for module in plan.modules:
        print(f"\n{module.name}: {module.description}")
        for t in plan.tasks:
            if t.module != module.name:
                continue
            deps = f" <- {', '.join(t.depends_on)}" if t.depends_on else ""
            print(f"  {t.id} [{t.size}, {t.priority:6}] {t.title}{deps}")

def print_validation(report: ValidationReport) -> None:
    if not report.issues:
        print("\nValidation: no issues")
        return
    print("\nValidation:")
    for issue in report.issues:
        print(f"  [{issue.level:7}] {issue.code}: {issue.message}")

def print_diff(diff: dict) -> None:
    print("\nChanges from the previous version:")
    for m in diff["added_modules"]:
        print(f"  + module {m}")
    for t in diff["added"]:
        print(f"  + {t}")
    for t in diff["removed"]:
        print(f"  - {t}")
    for c in diff["changed"]:
        print(f"  ~ {c['id']} ({', '.join(c['fields'])})")
    print(f"  {diff['unchanged']} tasks unchanged")


def show_review(state: dict, pending: dict, first: bool) -> None:
    if first:
        print("\n=== Requirements ===")
        print_requirements(state["requirements"])
        clarifications = state.get("clarifications", [])
        if clarifications:
            print("\nYour answers:")
        for c in clarifications:
            print(f"  Q: {c.question}\n  A: {c.answer}")
    print(f"\n=== Plan v{pending['plan_version']} ===")
    print_plan(state["plan"])
    print_validation(state["validation"])
    print("\n=== Schedule ===")
    print_schedule(state["plan"], state["schedule"])
    if pending.get("notice"):
        print(f"\nNote: {pending['notice']}")
    if pending["diff"] is not None:
        last = state["messages"][-1]
        if last.tool_calls:
            print("\nWarning: the agent hit its step limit before finishing. Some edits may be missing.")
        elif last.content:
            print(f"\nAgent: {last.content}")
        print_diff(pending["diff"])


def ask_review(pending: dict) -> dict:
    extra = ", 'd' to discard this version" if pending["can_discard"] else ""
    reply = input(f"\nEnter to approve{extra}, or type a change:\n> ").strip()
    while reply.lower() == "d" and not pending["can_discard"]:
        reply = input("\nThere is no earlier version to go back to. Enter to approve, or type a change:\n> ").strip()
    if not reply:
        return {"action": "approve"}
    if reply.lower() == "d" and pending["can_discard"]:
        return {"action": "discard"}
    return {"action": "change", "request": reply}


def ask_user(questions: list[str]) -> list[str]:
    print("\nI need a few answers before planning (press Enter to let me decide):")
    answers = []
    for n, question in enumerate(questions, start=1):
        answers.append(input(f"\n{n}. {question}\n> "))
    print()
    return answers

def run(idea: str) -> dict:
    graph = build_graph(make_llm(get_settings()), checkpointer=make_checkpointer(), human_review=True)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 60}
    payload: dict | Command = {"idea": idea}
    final: dict = {}
    start = time.perf_counter()
    first_review = True

    while True:
        pending = None
        for mode, chunk in graph.stream(payload, config, stream_mode=["updates", "values"]):
            if mode == "values":
                final = chunk
                continue
            if "__interrupt__" in chunk:
                pending = chunk["__interrupt__"][0].value
                continue
            for node, update in chunk.items():
                elapsed = time.perf_counter() - start
                for line in (update or {}).get("log", []):
                    print(f"[{elapsed:5.1f}s] {line}")

        if pending is None:
            return final
        if "error" in pending:
            print(f"\n{pending['error']}")
        if pending.get("kind") == "review":
            show_review(final, pending, first=first_review)
            first_review = False
            payload = Command(resume=ask_review(pending))
        else:
            payload = Command(resume=ask_user(pending["questions"]))

def main() -> None:
    idea = " ".join(sys.argv[1:]) or "A mobile app for college roommates to split expenses"
    try:
        state = run(idea)
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.", file=sys.stderr)
        sys.exit(130)
    except RuntimeError as e:
        sys.exit(f"Config error: {e}")
    except LLMOutputError as e:
        sys.exit(f"Bad model output: {e}")
    except groq.AuthenticationError:
        sys.exit("Invalid GROQ_API_KEY. Check your .env file.")
    except groq.RateLimitError:
        sys.exit("Rate limited by Groq. Wait a minute and retry.")
    except groq.APIConnectionError:
        sys.exit("Could not reach Groq. Check your internet connection.")
    except groq.APIStatusError as e:
        sys.exit(f"Groq returned HTTP {e.status_code}: {e.message}")

    if "schedule" in state:
        print(f"\nApproved plan v{state.get('plan_version', 1)}.")
        return
    print_plan(state["plan"])
    print_validation(state["validation"])
    print(f"\nPlan is still invalid after {MAX_REPAIR_ATTEMPTS} repair attempts.")


if __name__ == "__main__":
    main()

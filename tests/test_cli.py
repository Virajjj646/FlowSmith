"""Step 5: the CLI drives the interrupt/resume loop through input()."""

import sys

import pytest
from helpers import (
    ScriptedAgent, ScriptedLLM, agent_reply, blocking_req, make_plan, make_req, make_task,
    make_valid_plan, tool_calls,
)

from flowsmith import cli
from flowsmith import graph as graph_module
from flowsmith.graph import MAX_AGENT_STEPS, MAX_CLARIFY_ROUNDS
from flowsmith.schemas import Clarification

IDEA = "a todo app"


class ScriptedInput:
    """Replaces builtins.input: records each prompt, returns scripted answers."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f"input() called more often than scripted: {prompt!r}")
        return self.answers.pop(0)


@pytest.fixture
def patch_cli(monkeypatch):
    """patch_cli(requirements, input_fn, agent=None) -> ScriptedLLM. No settings, no network.

    call_agent is always patched: with no scripted agent, any revise step fails loudly.
    """

    def _patch(requirements, input_fn, agent=None, **llm_kwargs):
        fake_llm = ScriptedLLM(*requirements, **llm_kwargs)
        monkeypatch.setattr(cli, "get_settings", lambda: None)
        monkeypatch.setattr(cli, "make_llm", lambda settings: None)
        monkeypatch.setattr(graph_module, "call_structured", fake_llm)
        monkeypatch.setattr(graph_module, "call_agent", agent or ScriptedAgent())
        monkeypatch.setattr("builtins.input", input_fn)
        monkeypatch.setattr(sys, "argv", ["flowsmith", IDEA])
        return fake_llm

    return _patch


@pytest.fixture
def run_cli(patch_cli):
    """run_cli(requirements, answers) -> (final_state, ScriptedLLM, ScriptedInput)."""

    def _run(requirements, answers=(), agent=None, **llm_kwargs):
        fake_input = ScriptedInput(*answers)
        fake_llm = patch_cli(requirements, fake_input, agent, **llm_kwargs)
        return cli.run(IDEA), fake_llm, fake_input

    return _run


def raising(exc_type):
    def _input(prompt=""):
        raise exc_type()

    return _input


def test_cli_shows_every_question_in_input_prompts(run_cli):
    # Bug: ask_user skips/reorders questions, or the review prompt comes before (or instead of) them.
    _, _, fake_input = run_cli(
        [blocking_req("Platform?", "Payments?"), make_req()], answers=["Web", "Stripe", ""]
    )
    assert len(fake_input.prompts) == 3
    assert "Platform?" in fake_input.prompts[0]
    assert "Payments?" in fake_input.prompts[1]
    assert "Enter to approve" in fake_input.prompts[2]


def test_cli_prints_the_clarification_header(run_cli, capsys):
    # Bug: ask_user header dropped, so the user doesn't know Enter means "you decide".
    run_cli([blocking_req("Platform?"), make_req()], answers=["Web", ""])
    assert "I need a few answers before planning" in capsys.readouterr().out


def test_cli_answers_reach_the_graph(run_cli):
    # Bug: run() resumes with the payload instead of Command(resume=answers), or restarts the run.
    final, fake_llm, _ = run_cli(
        [blocking_req("Platform?", "Payments?"), make_req()], answers=["Web", "Stripe", ""]
    )
    assert "Q: Platform?\nA: Web" in fake_llm.analyze_inputs[1]
    assert "Q: Payments?\nA: Stripe" in fake_llm.analyze_inputs[1]
    assert final["clarifications"] == [
        Clarification(question="Platform?", answer="Web"),
        Clarification(question="Payments?", answer="Stripe"),
    ]


def test_cli_returns_final_state_with_schedule(run_cli):
    # Bug: run() returns after the first stream (the paused state) instead of looping.
    final, _, _ = run_cli([blocking_req("Platform?"), make_req()], answers=["Web", ""])
    assert "schedule" in final
    assert "plan" in final
    assert "__interrupt__" not in final


@pytest.mark.skipif(MAX_CLARIFY_ROUNDS < 2, reason="needs two clarification rounds")
def test_cli_handles_two_rounds_of_questions(run_cli):
    # Bug: `pending` not reset per stream, or the loop exits after one resume.
    final, _, fake_input = run_cli(
        [blocking_req("Platform?"), blocking_req("Payments?"), make_req()],
        answers=["Web", "Stripe", ""],
    )
    assert ["Platform?" in fake_input.prompts[0], "Payments?" in fake_input.prompts[1]] == [True, True]
    assert final["clarification_rounds"] == 2
    assert "schedule" in final


def test_cli_without_blocking_gaps_asks_no_clarification_question(run_cli):
    # Bug: ask_user called unconditionally (e.g. with an empty question list) before the review prompt.
    final, _, fake_input = run_cli([make_req()], answers=[""])
    assert len(fake_input.prompts) == 1
    assert "Enter to approve" in fake_input.prompts[0]
    assert "schedule" in final


def test_cli_prints_reask_error_before_asking_again(run_cli, monkeypatch, capsys):
    # Bug: run() ignores pending["error"], so the user is re-asked without being told why.
    real_ask_user = cli.ask_user
    replies = iter([["Web", "Extra"]])  # first reply has the wrong count, then fall back to input()
    monkeypatch.setattr(cli, "ask_user", lambda qs: next(replies, None) or real_ask_user(qs))
    final, _, _ = run_cli([blocking_req("Platform?"), make_req()], answers=["Web", ""])
    out = capsys.readouterr().out
    assert "Expected a list of 1 text answers, got ['Web', 'Extra']." in out
    assert final["clarifications"] == [Clarification(question="Platform?", answer="Web")]


@pytest.mark.parametrize("exc_type", [EOFError, KeyboardInterrupt], ids=["ctrl-d", "ctrl-c"])
def test_main_exits_130_when_input_is_cancelled(patch_cli, capsys, exc_type):
    # Bug: EOFError/KeyboardInterrupt not caught in main() -> traceback and exit code 1.
    patch_cli([blocking_req("Platform?")], raising(exc_type))
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 130


@pytest.mark.parametrize("exc_type", [EOFError, KeyboardInterrupt], ids=["ctrl-d", "ctrl-c"])
def test_main_prints_cancelled_without_traceback_when_input_is_cancelled(patch_cli, capsys, exc_type):
    # Bug: message goes to stdout / sys.exit(str) used / exception re-raised with a traceback.
    patch_cli([blocking_req("Platform?")], raising(exc_type))
    with pytest.raises(SystemExit):
        cli.main()
    captured = capsys.readouterr()
    assert captured.err.strip() == "Cancelled."
    assert "Traceback" not in captured.err + captured.out


# ---------------------------------------------------------------------------
# Step 6b: review in the CLI
# ---------------------------------------------------------------------------

def resize_t02_agent():
    return ScriptedAgent(
        tool_calls(("UpdateTask", {"task_id": "T02", "changes": {"size": "L"}})),
        agent_reply("Made T02 large."),
    )


def test_cli_first_review_shows_requirements_and_plan_v1(run_cli, capsys):
    # Bug: show_review not called before ask_review, so the user approves a plan they never saw.
    run_cli([make_req()], answers=[""])
    out = capsys.readouterr().out
    assert out.index("=== Requirements ===") < out.index("=== Plan v1 ===") < out.index("=== Schedule ===")


def test_cli_first_review_prompt_does_not_offer_discard(run_cli):
    # Bug: 'd' advertised on v1, where there is nothing to discard to.
    _, _, fake_input = run_cli([make_req()], answers=[""])
    assert "discard" not in fake_input.prompts[0]


def test_cli_typed_change_reaches_the_agent(run_cli):
    # Bug: ask_review returns approve for any text, or the request isn't forwarded to revise.
    agent = resize_t02_agent()
    run_cli([make_req()], answers=["Make T02 large", ""], agent=agent)
    assert "Change request: Make T02 large" in agent.calls[0][1][0].content


def test_cli_second_review_shows_agent_summary_and_diff(run_cli, capsys):
    # Bug: v2 review hides what changed, so the user can't judge the revision.
    run_cli([make_req()], answers=["Make T02 large", ""], agent=resize_t02_agent())
    out = capsys.readouterr().out
    assert "=== Plan v2 ===" in out
    assert "Agent: Made T02 large." in out
    assert "Changes from the previous version:\n  ~ T02 (size)\n  1 tasks unchanged" in out


def test_cli_second_review_prompt_offers_discard(run_cli):
    # Bug: can_discard ignored in the prompt, so users don't know they can undo.
    _, _, fake_input = run_cli([make_req()], answers=["Make T02 large", ""], agent=resize_t02_agent())
    assert "'d' to discard this version" in fake_input.prompts[1]


def test_cli_approving_revision_returns_v2(run_cli):
    # Bug: run() returns the paused/previous state instead of the approved revision.
    final, _, _ = run_cli([make_req()], answers=["Make T02 large", ""], agent=resize_t02_agent())
    assert final["plan_version"] == 2
    assert next(t for t in final["plan"].tasks if t.id == "T02").size == "L"


def test_cli_d_discards_revision_and_returns_to_v1(run_cli):
    # Bug: 'd' sent as a change request ("d") instead of a discard.
    final, _, _ = run_cli([make_req()], answers=["Make T02 large", "d", ""], agent=resize_t02_agent())
    assert final["plan_version"] == 1
    assert final["plan"] == make_valid_plan()


def test_cli_uppercase_d_also_discards(run_cli):
    # Bug: case-sensitive comparison -> "D" starts a new revision with request "D".
    final, _, _ = run_cli([make_req()], answers=["Make T02 large", "D", ""], agent=resize_t02_agent())
    assert final["plan_version"] == 1


def test_main_prints_approved_version(patch_cli, capsys):
    # Bug: main() still prints the old full report, or reports the wrong version.
    patch_cli([make_req()], ScriptedInput("Make T02 large", ""), resize_t02_agent())
    cli.main()
    assert capsys.readouterr().out.rstrip().endswith("Approved plan v2.")


# --- ask_review mapping ------------------------------------------------------

def review_pending(can_discard):
    return {"kind": "review", "plan_version": 2 if can_discard else 1, "diff": None, "can_discard": can_discard}


@pytest.mark.parametrize("can_discard", [False, True], ids=["v1", "v2"])
@pytest.mark.parametrize("reply", ["", "   "], ids=["empty", "spaces"])
def test_ask_review_blank_reply_approves(monkeypatch, reply, can_discard):
    # Bug: whitespace not stripped -> pressing space+Enter sends a change request "   ".
    monkeypatch.setattr("builtins.input", ScriptedInput(reply))
    assert cli.ask_review(review_pending(can_discard)) == {"action": "approve"}


@pytest.mark.parametrize("reply", ["d", "D", " d "], ids=["lower", "upper", "padded"])
def test_ask_review_d_discards_when_allowed(monkeypatch, reply):
    # Bug: case/whitespace-sensitive check -> "D" becomes a change request.
    monkeypatch.setattr("builtins.input", ScriptedInput(reply))
    assert cli.ask_review(review_pending(can_discard=True)) == {"action": "discard"}


def test_ask_review_d_on_v1_asks_again_instead_of_sending_a_change(monkeypatch):
    # Bug: "d" on v1 becomes change request "d", costing an LLM call for a keystroke.
    fake_input = ScriptedInput("d", "D", "")
    monkeypatch.setattr("builtins.input", fake_input)
    assert cli.ask_review(review_pending(can_discard=False)) == {"action": "approve"}
    assert len(fake_input.prompts) == 3
    assert all("no earlier version to go back to" in p for p in fake_input.prompts[1:])


def test_ask_review_text_is_a_stripped_change_request(monkeypatch):
    # Bug: raw reply forwarded with surrounding whitespace/newline into the agent prompt.
    monkeypatch.setattr("builtins.input", ScriptedInput("  Add billing  "))
    assert cli.ask_review(review_pending(can_discard=True)) == {"action": "change", "request": "Add billing"}


def test_cli_requirements_printed_only_on_first_review(run_cli, capsys):
    # Bug: show_review checks the wrong flag, so every revision re-prints the requirements.
    run_cli([make_req()], answers=["Make T02 large", ""], agent=resize_t02_agent())
    out = capsys.readouterr().out
    assert out.count("=== Requirements ===") == 1
    assert out.index("=== Requirements ===") < out.index("=== Plan v1 ===")


# --- Step 6b fixes: requirements once, notices, step-limit warning ------------

def cyclic_valid_plan():
    """make_valid_plan() with T01 <-> T02: what a failing repair keeps returning."""
    return make_plan(make_task("T01", depends_on=["T02"]), make_task("T02", depends_on=["T01"]))


def cycle_agent():
    return ScriptedAgent(
        tool_calls(("UpdateTask", {"task_id": "T01", "changes": {"depends_on": ["T02"]}})),
        agent_reply("T01 now waits for T02."),
    )


def test_cli_requirements_printed_once_across_change_rollback_approve(run_cli, capsys):
    # Bug: requirements keyed on diff-is-None, so the review after a rollback prints them again.
    final, _, _ = run_cli(
        [make_req()], answers=["T01 after T02", ""], agent=cycle_agent(),
        repair_plans=[cyclic_valid_plan()] * 2,
    )
    assert final["plan_version"] == 1  # precondition: it really rolled back
    assert capsys.readouterr().out.count("=== Requirements ===") == 1


def test_cli_requirements_printed_once_across_change_discard_approve(run_cli, capsys):
    # Bug: requirements keyed on diff-is-None, so the review after a discard prints them again.
    run_cli([make_req()], answers=["Make T02 large", "d", ""], agent=resize_t02_agent())
    assert capsys.readouterr().out.count("=== Requirements ===") == 1


def test_cli_prints_rollback_notice(run_cli, capsys):
    # Bug: notice in the payload but never shown, so a rollback looks like a fresh start.
    run_cli(
        [make_req()], answers=["T01 after T02", ""], agent=cycle_agent(),
        repair_plans=[cyclic_valid_plan()] * 2,
    )
    assert "Note: The change 'T01 after T02' produced a plan that could not be repaired" in capsys.readouterr().out


def test_cli_prints_discard_notice(run_cli, capsys):
    # Bug: notice printed only for rollbacks, not discards.
    run_cli([make_req()], answers=["Make T02 large", "d", ""], agent=resize_t02_agent())
    assert "Note: Discarded v2. Back to v1." in capsys.readouterr().out


def test_cli_warns_instead_of_agent_summary_at_step_limit(run_cli, capsys):
    # Bug: the empty "Agent:" line (or nothing) is shown when the agent was cut off mid-edit.
    agent = ScriptedAgent(*[tool_calls(("GetTask", {"task_id": "T01"})) for _ in range(MAX_AGENT_STEPS)])
    run_cli([make_req()], answers=["Keep going", ""], agent=agent)
    out = capsys.readouterr().out
    assert "Warning: the agent hit its step limit before finishing. Some edits may be missing." in out
    assert "Agent:" not in out

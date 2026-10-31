"""Step 5: human-in-the-loop clarification (interrupt / resume)."""

import logging
from typing import Annotated, get_args, get_origin, get_type_hints

import pytest
from helpers import ScriptedLLM, blocking_req, make_gap, make_req
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde import jsonplus
from langgraph.types import Command
from pydantic import BaseModel

from flowsmith import graph as graph_module
from flowsmith.checkpoint import STATE_TYPES, make_checkpointer
from flowsmith.graph import (
    MAX_CLARIFY_ROUNDS,
    AgentState,
    NO_PREFERENCE,
    blocking_questions,
    build_analyze_input,
    build_graph,
    route_after_analyze,
    valid_answers,
)
from flowsmith.scheduling import Schedule
from flowsmith.schemas import Clarification, Feature, Gap, Plan, Requirements
from flowsmith.validation import ValidationReport

IDEA = "a todo app"
SERDE_LOGGER = "langgraph.checkpoint.serde.jsonplus"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def config_for(thread_id):
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}


@pytest.fixture
def cfg(request):
    # Unique per test: the test's own node id.
    return config_for(request.node.nodeid)


@pytest.fixture
def make_app(monkeypatch):
    """make_app(*requirements, checkpointer=...) -> (compiled graph, ScriptedLLM)."""

    def _make(*requirements, checkpointer="default"):
        fake = ScriptedLLM(*requirements)
        monkeypatch.setattr(graph_module, "call_structured", fake)
        saver = make_checkpointer() if checkpointer == "default" else checkpointer
        return build_graph(llm=None, checkpointer=saver), fake

    return _make


def run_answering(app, cfg, *answer_rounds):
    """Start the run, then resume once per answer list. Returns the last result."""
    result = app.invoke({"idea": IDEA}, cfg)
    for answers in answer_rounds:
        result = app.invoke(Command(resume=answers), cfg)
    return result


# ---------------------------------------------------------------------------
# 1. route_after_analyze
# ---------------------------------------------------------------------------

def test_route_blocking_gaps_without_rounds_key_goes_to_clarify():
    # Bug: state["clarification_rounds"] instead of .get(...) -> KeyError on first analyze.
    assert route_after_analyze({"requirements": blocking_req("Q?")}) == "clarify"


def test_route_blocking_gaps_below_max_rounds_goes_to_clarify():
    # Bug: comparison against MAX-1 or `<` swapped for `==`, so the last allowed round is skipped.
    state = {"requirements": blocking_req("Q?"), "clarification_rounds": MAX_CLARIFY_ROUNDS - 1}
    assert route_after_analyze(state) == "clarify"


def test_route_minor_only_gaps_goes_to_decompose():
    # Bug: routing on `req.gaps` instead of blocking gaps -> user is asked trivia.
    req = make_req(gaps=[make_gap("Colour scheme?", "minor")])
    assert route_after_analyze({"requirements": req}) == "decompose"


def test_route_no_gaps_goes_to_decompose():
    # Bug: condition inverted, or clarify always taken on the first round.
    assert route_after_analyze({"requirements": make_req()}) == "decompose"


def test_route_blocking_gaps_at_max_rounds_goes_to_decompose():
    # Bug: `<=` instead of `<` -> one clarification round too many.
    state = {"requirements": blocking_req("Q?"), "clarification_rounds": MAX_CLARIFY_ROUNDS}
    assert route_after_analyze(state) == "decompose"


# ---------------------------------------------------------------------------
# 2. blocking_questions
# ---------------------------------------------------------------------------

def test_blocking_questions_returns_only_blocking_in_order():
    # Bug: minor gaps leak through, or questions are sorted/deduplicated via a set.
    req = make_req(
        gaps=[
            make_gap("Zebra platform?"),
            make_gap("Dark mode?", "minor"),
            make_gap("Apple payments?"),
            make_gap("Font?", "minor"),
        ]
    )
    assert blocking_questions(req) == ["Zebra platform?", "Apple payments?"]


# ---------------------------------------------------------------------------
# 3. build_analyze_input
# ---------------------------------------------------------------------------

def test_analyze_input_without_clarifications_key_is_idea_unchanged():
    # Bug: header appended unconditionally, or state["clarifications"] -> KeyError.
    assert build_analyze_input({"idea": IDEA}) == IDEA


def test_analyze_input_with_empty_clarifications_is_idea_unchanged():
    # Bug: `is not None` check instead of truthiness -> empty "Answers" header sent to the LLM.
    assert build_analyze_input({"idea": IDEA, "clarifications": []}) == IDEA


def test_analyze_input_appends_every_qa_pair_in_order():
    # Bug: only the latest round/pair is sent, or pairs are reordered.
    state = {
        "idea": IDEA,
        "clarifications": [
            Clarification(question="Platform?", answer="Web"),
            Clarification(question="Payments?", answer="No"),
        ],
    }
    assert build_analyze_input(state) == (
        f"{IDEA}\n\nAnswers to your earlier questions:\n"
        "Q: Platform?\nA: Web\n"
        "Q: Payments?\nA: No"
    )


# ---------------------------------------------------------------------------
# 4. No blocking gaps
# ---------------------------------------------------------------------------

def test_no_blocking_gaps_runs_to_schedule_without_pausing(make_app, cfg):
    # Bug: clarify is wired as a plain edge, so every run pauses.
    app, _ = make_app(make_req())
    result = app.invoke({"idea": IDEA}, cfg)
    assert "__interrupt__" not in result
    assert "schedule" in result
    assert app.get_state(cfg).next == ()


def test_minor_only_gaps_run_to_schedule_without_pausing(make_app, cfg):
    # Bug: graph routes on any gap, so minor questions interrupt the user.
    app, _ = make_app(make_req(gaps=[make_gap("Dark mode?", "minor")]))
    result = app.invoke({"idea": IDEA}, cfg)
    assert "__interrupt__" not in result
    assert "schedule" in result


# ---------------------------------------------------------------------------
# 5. Blocking gaps pause the run
# ---------------------------------------------------------------------------

def test_blocking_gaps_invoke_returns_interrupt(make_app, cfg):
    # Bug: clarify doesn't call interrupt() (e.g. reads answers from state) -> run never pauses.
    app, _ = make_app(blocking_req("Platform?"))
    result = app.invoke({"idea": IDEA}, cfg)
    assert "__interrupt__" in result


def test_blocking_gaps_leave_thread_paused_at_clarify(make_app, cfg):
    # Bug: interrupt raised from analyze instead of clarify -> resume re-runs the LLM call.
    app, _ = make_app(blocking_req("Platform?"))
    app.invoke({"idea": IDEA}, cfg)
    assert app.get_state(cfg).next == ("clarify",)


def test_interrupt_payload_lists_exactly_the_blocking_questions(make_app, cfg):
    # Bug: payload built from req.gaps (includes minor) or passes Gap objects instead of strings.
    req = make_req(
        gaps=[make_gap("Platform?"), make_gap("Dark mode?", "minor"), make_gap("Payments?")]
    )
    app, _ = make_app(req)
    result = app.invoke({"idea": IDEA}, cfg)
    assert [i.value for i in result["__interrupt__"]] == [
        {"questions": ["Platform?", "Payments?"]}
    ]


def test_paused_run_has_not_started_decompose(make_app, cfg):
    # Bug: analyze also has a static edge to decompose, so planning runs before the answers.
    app, fake = make_app(blocking_req("Platform?"))
    app.invoke({"idea": IDEA}, cfg)
    assert fake.plan_calls == []


# ---------------------------------------------------------------------------
# 6. Resume
# ---------------------------------------------------------------------------

def test_resume_completes_run_with_schedule(make_app, cfg):
    # Bug: clarify has no outgoing edge, so the run ends right after the answers.
    app, _ = make_app(blocking_req("Platform?"), make_req())
    result = run_answering(app, cfg, ["Web"])
    assert "__interrupt__" not in result
    assert "schedule" in result
    assert app.get_state(cfg).next == ()


def test_resume_stores_clarification_objects_with_qa_pairs(make_app, cfg):
    # Bug: answers zipped against the wrong list, or stored as raw strings/dicts.
    app, _ = make_app(blocking_req("Platform?", "Payments?"), make_req())
    result = run_answering(app, cfg, ["Web", "Stripe"])
    assert result["clarifications"] == [
        Clarification(question="Platform?", answer="Web"),
        Clarification(question="Payments?", answer="Stripe"),
    ]
    assert all(type(c) is Clarification for c in result["clarifications"])


def test_resume_sets_clarification_rounds_to_one(make_app, cfg):
    # Bug: counter not returned from clarify, or starts at 1 and increments to 2.
    app, _ = make_app(blocking_req("Platform?"), make_req())
    result = run_answering(app, cfg, ["Web"])
    assert result["clarification_rounds"] == 1


def test_second_analyze_call_contains_the_answers(make_app, cfg):
    # Bug: analyze still sends state["idea"], so the LLM never sees the answers.
    app, fake = make_app(blocking_req("Platform?", "Payments?"), make_req())
    run_answering(app, cfg, ["Web only", "Stripe"])
    assert len(fake.analyze_inputs) == 2
    second = fake.analyze_inputs[1]
    assert IDEA in second
    assert "Q: Platform?\nA: Web only" in second
    assert "Q: Payments?\nA: Stripe" in second


def test_first_analyze_call_is_the_bare_idea(make_app, cfg):
    # Bug: an empty "Answers to your earlier questions" header is sent before any answers.
    app, fake = make_app(blocking_req("Platform?"), make_req())
    run_answering(app, cfg, ["Web"])
    assert fake.analyze_inputs[0] == IDEA


# ---------------------------------------------------------------------------
# 7. Empty answers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blank", ["", "   ", "\t \n"], ids=["empty", "spaces", "mixed-ws"])
def test_blank_answer_is_stored_as_no_preference(make_app, cfg, blank):
    # Bug: `or NO_PREFERENCE` missing, or applied before strip() -> whitespace stored as-is.
    app, _ = make_app(blocking_req("Platform?"), make_req())
    result = run_answering(app, cfg, [blank])
    assert result["clarifications"] == [Clarification(question="Platform?", answer=NO_PREFERENCE)]


def test_answer_surrounding_whitespace_is_stripped(make_app, cfg):
    # Bug: strip() result only used for the emptiness check, raw answer stored.
    app, _ = make_app(blocking_req("Platform?"), make_req())
    result = run_answering(app, cfg, ["  Web  \n"])
    assert result["clarifications"][0].answer == "Web"


# ---------------------------------------------------------------------------
# 8. Two rounds accumulate
# ---------------------------------------------------------------------------

@pytest.mark.skipif(MAX_CLARIFY_ROUNDS < 2, reason="needs two clarification rounds")
def test_two_rounds_accumulate_clarifications(make_app, cfg):
    # Bug: clarifications declared without the operator.add reducer -> round 2 overwrites round 1.
    app, _ = make_app(blocking_req("Platform?"), blocking_req("Payments?"), make_req())
    result = run_answering(app, cfg, ["Web"], ["Stripe"])
    assert result["clarifications"] == [
        Clarification(question="Platform?", answer="Web"),
        Clarification(question="Payments?", answer="Stripe"),
    ]
    assert result["clarification_rounds"] == 2


@pytest.mark.skipif(MAX_CLARIFY_ROUNDS < 2, reason="needs two clarification rounds")
def test_third_analyze_call_sees_answers_from_both_rounds(make_app, cfg):
    # Bug: analyze input built from the latest round only, so the LLM re-asks round-1 questions.
    app, fake = make_app(blocking_req("Platform?"), blocking_req("Payments?"), make_req())
    run_answering(app, cfg, ["Web"], ["Stripe"])
    third = fake.analyze_inputs[2]
    assert "Q: Platform?\nA: Web\nQ: Payments?\nA: Stripe" in third


# ---------------------------------------------------------------------------
# 9. Clarify loop is bounded
# ---------------------------------------------------------------------------

def test_always_blocking_pauses_exactly_max_rounds_then_decomposes(make_app, cfg):
    # Bug: round counter never incremented -> pauses forever / hits the recursion limit.
    reqs = [blocking_req(f"Question {n}?") for n in range(MAX_CLARIFY_ROUNDS + 1)]
    app, fake = make_app(*reqs)

    pauses = 0
    result = app.invoke({"idea": IDEA}, cfg)
    while "__interrupt__" in result:
        pauses += 1
        assert pauses <= MAX_CLARIFY_ROUNDS, "paused more than MAX_CLARIFY_ROUNDS times"
        result = app.invoke(Command(resume=["whatever"]), cfg)

    assert pauses == MAX_CLARIFY_ROUNDS
    assert result["clarification_rounds"] == MAX_CLARIFY_ROUNDS
    assert len(fake.analyze_inputs) == MAX_CLARIFY_ROUNDS + 1
    assert len(fake.plan_calls) == 1
    assert "schedule" in result


# ---------------------------------------------------------------------------
# 10. Wrong number of answers
# ---------------------------------------------------------------------------

def interrupt_values(result):
    return [i.value for i in result.get("__interrupt__", [])]


# --- valid_answers (unit) --------------------------------------------------

def test_valid_answers_accepts_one_string_per_question():
    # Bug: check inverted or too strict (e.g. rejects empty strings, which mean "no preference").
    assert valid_answers(["Web", ""], ["Platform?", "Payments?"]) is True


def test_valid_answers_rejects_wrong_length():
    # Bug: length check missing -> zip() silently drops answers or questions.
    assert valid_answers(["Web"], ["Platform?", "Payments?"]) is False


def test_valid_answers_rejects_non_list():
    # Bug: isinstance(list) check missing -> a 1-char string passes as a 1-answer list.
    assert valid_answers("W", ["Platform?"]) is False


def test_valid_answers_rejects_non_string_item():
    # Bug: item type check missing -> None reaches a.strip() and crashes clarify.
    assert valid_answers([None], ["Platform?"]) is False


# --- re-asking in the graph -------------------------------------------------

def test_wrong_answer_count_does_not_raise(make_app, cfg):
    # Bug: clarify still raises on bad input, which used to leave the thread stuck forever.
    app, _ = make_app(blocking_req("Platform?"), make_req())
    app.invoke({"idea": IDEA}, cfg)
    app.invoke(Command(resume=["Web", "Extra"]), cfg)


def test_wrong_answer_count_pauses_again_at_clarify(make_app, cfg):
    # Bug: bad answers are accepted (zip truncates) and the run moves on to decompose.
    app, fake = make_app(blocking_req("Platform?"), make_req())
    app.invoke({"idea": IDEA}, cfg)
    app.invoke(Command(resume=["Web", "Extra"]), cfg)
    assert app.get_state(cfg).next == ("clarify",)
    assert fake.plan_calls == []


def test_reask_payload_has_error_and_same_questions(make_app, cfg):
    # Bug: re-ask payload drops the questions or omits the error, so the user can't tell what failed.
    app, _ = make_app(blocking_req("Platform?", "Payments?"), make_req())
    app.invoke({"idea": IDEA}, cfg)
    result = app.invoke(Command(resume=["Web"]), cfg)
    [payload] = interrupt_values(result)
    assert payload["questions"] == ["Platform?", "Payments?"]
    assert "Expected a list of 2 text answers" in payload["error"]
    assert "['Web']" in payload["error"]


def test_reask_does_not_call_analyze_again(make_app, cfg):
    # Bug: re-ask implemented by routing back to analyze, wasting an LLM call per typo.
    app, fake = make_app(blocking_req("Platform?"), make_req())
    app.invoke({"idea": IDEA}, cfg)
    app.invoke(Command(resume=["Web", "Extra"]), cfg)
    assert len(fake.analyze_inputs) == 1


def test_correct_resume_after_wrong_answer_count_completes(make_app, cfg):
    # Bug: the rejected resume value is replayed on the next resume, so the thread never recovers.
    app, _ = make_app(blocking_req("Platform?"), make_req())
    app.invoke({"idea": IDEA}, cfg)
    app.invoke(Command(resume=["Web", "Extra"]), cfg)
    result = app.invoke(Command(resume=["Web"]), cfg)
    assert "schedule" in result
    assert result["clarifications"] == [Clarification(question="Platform?", answer="Web")]
    assert result["clarification_rounds"] == 1


def test_plain_string_resume_is_reasked(make_app, cfg):
    # Bug: a bare string passes validation and is zipped character by character.
    app, _ = make_app(blocking_req("Platform?"), make_req())
    app.invoke({"idea": IDEA}, cfg)
    result = app.invoke(Command(resume="Web"), cfg)
    [payload] = interrupt_values(result)
    assert payload["questions"] == ["Platform?"]
    assert "'Web'" in payload["error"]


def test_non_string_item_resume_is_reasked(make_app, cfg):
    # Bug: item type not checked -> None.strip() raises AttributeError inside clarify.
    app, _ = make_app(blocking_req("Platform?"), make_req())
    app.invoke({"idea": IDEA}, cfg)
    result = app.invoke(Command(resume=[None]), cfg)
    assert [p["questions"] for p in interrupt_values(result)] == [["Platform?"]]


def test_two_bad_resumes_then_good_one_completes(make_app, cfg):
    # Bug: `if` instead of `while` -> the second bad answer is accepted without re-checking.
    app, _ = make_app(blocking_req("Platform?"), make_req())
    app.invoke({"idea": IDEA}, cfg)

    first = app.invoke(Command(resume=["Web", "Extra"]), cfg)
    second = app.invoke(Command(resume=[]), cfg)
    final = app.invoke(Command(resume=["Web"]), cfg)

    assert "error" in interrupt_values(first)[0]
    assert "got []" in interrupt_values(second)[0]["error"]
    assert "schedule" in final
    assert final["clarifications"] == [Clarification(question="Platform?", answer="Web")]


# ---------------------------------------------------------------------------
# 11. Thread isolation
# ---------------------------------------------------------------------------

def test_two_threads_do_not_share_state(make_app, cfg):
    # Bug: thread_id ignored (e.g. hard-coded config), so thread B resumes A's interrupt.
    cfg_a = config_for(cfg["configurable"]["thread_id"] + "/A")
    cfg_b = config_for(cfg["configurable"]["thread_id"] + "/B")
    # Call order: A analyze, B analyze, B re-analyze, A re-analyze.
    app, _ = make_app(blocking_req("A question?"), blocking_req("B question?"), make_req(), make_req())

    app.invoke({"idea": "idea A"}, cfg_a)
    app.invoke({"idea": "idea B"}, cfg_b)
    result_b = app.invoke(Command(resume=["B answer"]), cfg_b)

    state_a = app.get_state(cfg_a)
    assert state_a.next == ("clarify",)
    assert state_a.values["idea"] == "idea A"
    assert [i.value for i in state_a.interrupts] == [{"questions": ["A question?"]}]

    result_a = app.invoke(Command(resume=["A answer"]), cfg_a)
    assert result_a["clarifications"] == [Clarification(question="A question?", answer="A answer")]
    assert result_b["clarifications"] == [Clarification(question="B question?", answer="B answer")]
    assert result_b["idea"] == "idea B"


# ---------------------------------------------------------------------------
# 12-13. Checkpointer serializer
# ---------------------------------------------------------------------------

def test_checkpointed_state_holds_pydantic_instances(make_app, cfg):
    # Bug: serializer falls back to JSON/dicts -> state["plan"].tasks raises AttributeError.
    app, _ = make_app(blocking_req("Platform?"), make_req())
    run_answering(app, cfg, ["Web"])
    values = app.get_state(cfg).values
    assert type(values["requirements"]) is Requirements
    assert type(values["plan"]) is Plan
    assert type(values["schedule"]) is Schedule
    assert type(values["validation"]) is ValidationReport
    assert all(type(c) is Clarification for c in values["clarifications"])


@pytest.fixture
def serde_warnings(monkeypatch, caplog):
    """Capture LangGraph's serializer warnings, with its warn-once memory reset.

    The warning is a logging record (not warnings.warn), so caplog sees it and recwarn
    does not. LangGraph logs it only once per type per *process* (module-level set
    _warned_unregistered_types), so without this reset whichever test deserializes a type
    first swallows the warning and the assertions below become order-dependent.
    """
    monkeypatch.setattr(jsonplus, "_warned_unregistered_types", set())
    caplog.set_level(logging.WARNING, logger=SERDE_LOGGER)
    return caplog


def test_bare_in_memory_saver_logs_unregistered_type_warning(make_app, cfg, serde_warnings):
    # Control for the next test. Bug in the *test setup*: if caplog stops capturing this
    # warning (LangGraph moves it to warnings.warn or renames it), the next test passes vacuously.
    caplog = serde_warnings
    app, _ = make_app(blocking_req("Platform?"), make_req(), checkpointer=InMemorySaver())
    run_answering(app, cfg, ["Web"])
    app.get_state(cfg)
    assert any("unregistered type" in r.getMessage() for r in caplog.records)


def test_make_checkpointer_logs_no_unregistered_type_warning(make_app, cfg, serde_warnings):
    # Bug: a state type (e.g. Clarification or Schedule) missing from STATE_TYPES, or serde not passed.
    caplog = serde_warnings
    app, _ = make_app(blocking_req("Platform?"), make_req())
    run_answering(app, cfg, ["Web"])
    app.get_state(cfg)
    unregistered = [r.getMessage() for r in caplog.records if "unregistered type" in r.getMessage()]
    assert unregistered == []


# ---------------------------------------------------------------------------
# 14. No checkpointer
# ---------------------------------------------------------------------------

# What LangGraph (1.2.x) actually does: with no checkpointer, interrupt() does NOT raise.
# invoke() returns normally with the partial state plus "__interrupt__". The run simply
# stops at clarify; nothing is saved, so there is nothing to resume. A follow-up
# Command(resume=...) raises RuntimeError. The risk: a caller who forgets the checkpointer
# gets a silent, half-finished result (requirements but no plan) instead of an error.

def test_without_checkpointer_interrupt_returns_partial_state(make_app):
    # Bug: someone assumes this raises and wraps it in try/except, hiding unfinished runs.
    app, fake = make_app(blocking_req("Platform?"), checkpointer=None)
    result = app.invoke({"idea": IDEA}, {"recursion_limit": 30})
    assert [i.value for i in result["__interrupt__"]] == [{"questions": ["Platform?"]}]
    assert "requirements" in result
    assert "plan" not in result
    assert fake.plan_calls == []


def test_without_checkpointer_resume_raises_runtime_error(make_app):
    # Bug: resume silently starts a fresh run (re-calling the LLM) instead of failing loudly.
    app, _ = make_app(blocking_req("Platform?"), make_req(), checkpointer=None)
    app.invoke({"idea": IDEA}, {"recursion_limit": 30})
    with pytest.raises(RuntimeError, match="without checkpointer"):
        app.invoke(Command(resume=["Web"]), {"recursion_limit": 30})


# ---------------------------------------------------------------------------
# 15. Unanswered blocking questions are visible
# ---------------------------------------------------------------------------

def test_decompose_logs_unanswered_blocking_questions_after_max_rounds(make_app, cfg):
    # Bug: planning silently proceeds with open blocking questions after the round limit.
    reqs = [blocking_req(f"Question {n}?") for n in range(MAX_CLARIFY_ROUNDS)]
    reqs.append(blocking_req("Still open A?", "Still open B?"))
    app, _ = make_app(*reqs)
    result = run_answering(app, cfg, *[["answer"]] * MAX_CLARIFY_ROUNDS)
    assert "decompose: planning with 2 unanswered blocking questions" in result["log"]


def test_decompose_does_not_log_unanswered_when_all_answered(make_app, cfg):
    # Bug: the warning line is emitted unconditionally (or counts minor gaps).
    app, _ = make_app(blocking_req("Platform?"), make_req(gaps=[make_gap("Font?", "minor")]))
    result = run_answering(app, cfg, ["Web"])
    assert not any("unanswered" in line for line in result["log"])


# ---------------------------------------------------------------------------
# 16. STATE_TYPES covers every model that can end up in a checkpoint
# ---------------------------------------------------------------------------

def collect_models(tp, found=None):
    """Every pydantic BaseModel subclass reachable from a type hint, recursively.

    Unwraps Annotated/list/dict/Union/Optional via get_args, and descends into each
    model's own field annotations. Non-type args (Literal values, Annotated metadata
    such as operator.add) are skipped.
    """
    found = set() if found is None else found
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        if tp in found:
            return found
        found.add(tp)
        for field in tp.model_fields.values():
            collect_models(field.annotation, found)
        return found
    if get_origin(tp) is Annotated:
        return collect_models(get_args(tp)[0], found)
    for arg in get_args(tp):
        if isinstance(arg, type) or get_origin(arg) is not None:
            collect_models(arg, found)
    return found


def test_collect_models_finds_nested_models():
    # Bug in the walker itself: it stops at the top level and would let test_state_types_complete pass vacuously.
    assert collect_models(Annotated[list[Requirements], "meta"]) == {Requirements, Feature, Gap}


def test_state_types_complete():
    # Bug: a new state model (or a nested one, e.g. Milestone) is not added to STATE_TYPES -> comes back as a dict.
    hints = get_type_hints(AgentState, include_extras=True)
    models = set()
    for hint in hints.values():
        collect_models(hint, models)
    models = {m for m in models if m.__module__.startswith("flowsmith")}
    missing = sorted(m.__name__ for m in models - set(STATE_TYPES))
    assert missing == []
    assert models  # the walk found something

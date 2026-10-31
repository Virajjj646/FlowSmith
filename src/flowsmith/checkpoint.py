from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from flowsmith.scheduling import Milestone, Schedule
from flowsmith.schemas import Clarification, Feature, Gap, Module, Plan, Requirements, Task
from flowsmith.validation import Issue, ValidationReport

STATE_TYPES = [
    Requirements, Feature, Gap, Clarification,
    Plan, Module, Task,
    ValidationReport, Issue,
    Schedule, Milestone,
]


def make_checkpointer() -> InMemorySaver:
    serde = JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES)
    return InMemorySaver(serde=serde)

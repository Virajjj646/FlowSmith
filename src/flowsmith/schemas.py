from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Feature(StrictModel):
    id: str = Field(description="Sequential id: F1, F2, F3, ...")
    name: str
    description: str
    source: Literal["stated", "inferred"] = Field(
        description="'stated' if the user explicitly asked for it, 'inferred' if you added it"
    )

class Gap(StrictModel):
    question: str = Field(description="A question to ask the user")
    severity: Literal["blocking", "minor"] = Field(
        description="'blocking' if the answer would change architecture, scope or tech stack"
    )
    reason: str = Field(description="Why this matters for planning")

class Requirements(StrictModel):
    app_name: str = Field(description="Short working name for the app")
    summary: str = Field(description="One-sentence description of the app")
    platform: Literal["web", "mobile", "web_and_mobile", "unknown"]
    target_users: list[str]
    core_features: list[Feature]
    constraints: list[str] = Field(description="Limits the user stated: budget, tech, deadline")
    assumptions: list[str] = Field(description="Reasonable defaults you chose for minor gaps")
    gaps: list[Gap]

    @field_validator("core_features")
    @classmethod
    def at_least_one_feature(cls, v: list[Feature]) -> list[Feature]:
        if not v:
            raise ValueError("must contain at least one feature")
        return v

class Module(StrictModel):
    name: str
    description: str

class Task(StrictModel):
    id: str = Field(description="Unique id: T01, T02, ...")
    title: str
    description: str
    module: str = Field(description="Exact name of the module this task belongs to")
    size: Literal["S", "M", "L"] = Field(description="S: under a day, M: 1-3 days, L: 3-5 days")
    priority: Literal["must", "should", "could"]
    depends_on: list[str] = Field(description="Ids of the tasks that must be finished first")
    covers: list[str] = Field(description="Ids of the features this task implements")
    acceptance_criteria: list[str]

class Plan(StrictModel):
    modules: list[Module]
    tasks: list[Task]
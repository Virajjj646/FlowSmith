from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel
from flowsmith.config import Settings

class LLMOutputError(Exception):
    """The model replied, but the reply could nit be turned into our schema."""

def make_llm(settings: Settings) -> ChatGroq:
    return ChatGroq(
        model=settings.model,
        temperature=settings.temperature,
        api_key=settings.groq_api_key,
        max_tokens=8192,
        reasoning_effort="low",
        max_retries=2,
        timeout=90.0,
    )

def call_structured[T: BaseModel](llm: ChatGroq, schema: type[T], system: str, user: str) -> T:
    runnable = llm.with_structured_output(
        schema, method="json_schema", strict=True, include_raw=True
    )
    result = runnable.invoke([("system", system), ("human", user)])

    raw = result["raw"]
    if raw.response_metadata.get("finish_reason") == "length":
        raise LLMOutputError(f"{schema.__name__}: output was cut off before the JSON was complete")
    if result["parsing_error"]:
        raise LLMOutputError(f"{schema.__name__}: {result['parsing_error']}")
    return result["parsed"]

def call_agent(
    llm: ChatGroq, system: str, messages: list[BaseMessage], tools: list[type[BaseModel]]
) -> AIMessage:
    return llm.bind_tools(tools).invoke([SystemMessage(content=system), *messages])

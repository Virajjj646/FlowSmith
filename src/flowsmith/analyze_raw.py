import json
import sys

import groq
from groq import Groq
from pydantic import ValidationError

from flowsmith.config import get_settings
from flowsmith.schemas import Requirements
from flowsmith.prompts import REQUIREMENTS_SYSTEM


class LLMOutputError(Exception):
    """The model replied, but the reply could not be turned into our schema."""


def extract_requirements(client: Groq, model: str, idea: str) -> Requirements:
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": REQUIREMENTS_SYSTEM},
            {"role": "user", "content": idea},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "requirements",
                "strict": True,
                "schema": Requirements.model_json_schema(),
            },
        },
    )

    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise LLMOutputError("Output was cut off before the JSON was complete.")
    if not choice.message.content:
        raise LLMOutputError("Model returned an empty response.")

    try:
        return Requirements.model_validate(json.loads(choice.message.content))
    except json.JSONDecodeError as e:
        raise LLMOutputError(f"Response was not valid JSON: {e}") from e
    except ValidationError as e:
        raise LLMOutputError(f"JSON did not match the schema:\n{e}") from e


def print_requirements(req: Requirements) -> None:
    print(f"{req.app_name}: {req.summary}")
    print(f"Platform: {req.platform}")
    print(f"Users:    {', '.join(req.target_users)}")
    print("\nFeatures:")
    for f in req.core_features:
        print(f"  [{f.source:8}] {f.name}: {f.description}")
    if req.constraints:
        print("\nConstraints:")
        for c in req.constraints:
            print(f"  - {c}")
    if req.assumptions:
        print("\nAssumptions:")
        for a in req.assumptions:
            print(f"  - {a}")
    if req.gaps:
        print("\nGaps:")
        for g in req.gaps:
            print(f"  [{g.severity:8}] {g.question}  ({g.reason})")


def main() -> None:
    try:
        settings = get_settings()
    except RuntimeError as e:
        sys.exit(f"Config error: {e}")

    client = Groq(api_key=settings.groq_api_key, max_retries=2, timeout=60.0)
    idea = " ".join(sys.argv[1:]) or "A mobile app for college roommates to split expenses"

    try:
        req = extract_requirements(client, settings.model, idea)
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

    print_requirements(req)
    print("\n--- raw validated object ---")
    print(req.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
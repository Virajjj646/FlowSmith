import sys
import groq 
from langchain_groq import ChatGroq
from flowsmith.analyze_raw import print_requirements
from flowsmith.config import get_settings
from flowsmith.prompts import REQUIREMENTS_SYSTEM
from flowsmith.schemas import Requirements

def build_analyzer(llm: ChatGroq):
    return llm.with_structured_output(
        Requirements,
        method="json_schema",
        strict=True,
        include_raw=True,
    )

def main() -> None:
    try:
        settings = get_settings()
    except RuntimeError as e:
        sys.exit(f"Config error: {e}")

    llm = ChatGroq(
        model=settings.model,
        temperature=settings.temperature,
        api_key=settings.groq_api_key,
        max_retries=2,
        timeout=60.0,
    )
    analyzer = build_analyzer(llm)
    idea = " ".join(sys.argv[1:]) or "A mobile app for college roommates to split expenses"

    try:
        result = analyzer.invoke([("system", REQUIREMENTS_SYSTEM), ("human", idea)])
    except groq.AuthenticationError:
        sys.exit("Invalid GROQ_API_KEY. Check your .env file.")
    except groq.RateLimitError:
        sys.exit("Rate limited by Groq. Wait a minute and retry.")
    except groq.APIConnectionError:
        sys.exit("Could not reach Groq. Check your internet connection.")
    except groq.APIStatusError as e:
        sys.exit(f"Groq returned HTTP {e.status_code}: {e.message}")

    if result["parsing_error"]:
        sys.exit(f"Bad model output: {result['parsing_error']}")

    print_requirements(result["parsed"])

    raw = result["raw"]
    print("\n--- what LangChain hid from you ---")
    print(f"raw type       {type(raw).__name__}")
    print(f"finish_reason  {raw.response_metadata.get('finish_reason')}")
    print(f"tokens         {raw.usage_metadata}")


if __name__ == "__main__":
    main()    
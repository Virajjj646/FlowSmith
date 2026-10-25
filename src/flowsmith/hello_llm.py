import sys
import time

import groq
from groq import Groq

from flowsmith.config import get_settings

SYSTEM_PROMPT = "You are a senior software architect. Answer concisely."

def main() -> None:
    try:
        settings = get_settings()
    except RuntimeError as e:
        sys.exit(f"Config error: {e}")

    client = Groq(api_key = settings.groq_api_key, max_retries=2, timeout=30.0)
    idea = " ".join(sys.argv[1:]) or "An app for roommates to split shared expenses"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"List the 5 core features of this app idea:\n{idea}"},
    ]

    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=settings.model,
            messages=messages,
            temperature= settings.temperature,
        )
    except groq.AuthenticationError:
        sys.exit("Invalid GROQ_API_KEY. Check your .env file")
    except groq.RateLimitError:
        sys.exit("Rate limited by Groq. Wait a minute and retry.")
    except groq.APIConnectionError:
        sys.exit("Could not reach Groq. Check your internet connection.")
    except groq.APIStatusError as e:
        sys.exit(f"Groq returned HTTP {e.status_code}: {e.message}")
    elapsed = time.perf_counter() - start

    choice = response.choices[0]
    usage = response.usage
    print(choice.message.content)
    print("\n--- response metadata ---")
    print(f"model          {response.model}")
    print(f"finish_reason  {choice.finish_reason}")
    print(f"tokens         {usage.prompt_tokens} in / {usage.completion_tokens} out")
    print(f"latency        {elapsed:.2f}s")

if __name__ == "__main__":
    main()
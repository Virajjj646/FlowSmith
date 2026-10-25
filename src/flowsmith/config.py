import os
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    model: str
    temperature: float

def get_settings() -> Settings:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    return Settings(
        groq_api_key=key,
        model=os.getenv("FLOWSMITH_MODEL", "openai/gpt-oss-120b"),
        temperature=float(os.getenv("FLOWSMITH_TEMPERATURE", "0.2")),
    )
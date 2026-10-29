from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    hf_token: str | None
    hf_space: str
    cors_origins: tuple[str,...]
    registry_path: Path

    @classmethod
    def from_env(cls):
        return cls(groq_api_key=os.getenv('GROQ_API_KEY',''),
                   groq_model=os.getenv('GROQ_MODEL','llama-3.3-70b-versatile'),
                   hf_token=os.getenv('HF_TOKEN') or None,
                   hf_space=os.getenv('HF_SPACE','adamnouassida5/deeplearning'),
                   cors_origins=tuple(x.strip() for x in os.getenv('CORS_ORIGINS','http://localhost:5173').split(',') if x.strip()),
                   registry_path=ROOT/'models/registry.json')

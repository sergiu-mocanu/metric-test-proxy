from dataclasses import dataclass
from typing import Protocol

@dataclass
class GenerationConfig:
    temperature: float = 0.2
    max_tokens: int = 256
    top_p: float = 0.95
    stop: list[str] | None = None


class CodeGenerator(Protocol):
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        ...
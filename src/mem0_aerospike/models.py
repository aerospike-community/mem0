from dataclasses import dataclass


@dataclass
class MemoryResult:
    id: str
    payload: dict
    score: float | None = None

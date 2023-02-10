from dataclasses import dataclass


@dataclass
class LocalStateRef:
    key: str
    init: bool = False

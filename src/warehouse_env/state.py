from dataclasses import dataclass
from typing import Tuple, List

# A grid coordinate stored as (x, y)
Pos = Tuple[int, int]

@dataclass
class Package:
    id: int
    pos: Pos
    delivered: bool = False

@dataclass
class EnvState:
    step_count: int
    agent_pos: Pos
    carrying_id: int | None
    battery: int
    packages: List[Package]
    walls: set[Pos]

    @property
    def is_carrying(self) -> bool:
        return self.carrying_id is not None
from dataclasses import dataclass
from typing import Tuple, List

# A grid coordinate stored as (x, y)
Pos = Tuple[int, int]

@dataclass
class Package:
    pos: Pos
    delivered: bool = False

@dataclass
class EnvState:
    step_count: int
    agent_pos: Pos
    carrying: bool
    battery: int
    packages: List[Package]

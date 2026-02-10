from dataclasses import dataclass

@dataclass(frozen=True)
class EnvConfig:
    width: int = 7
    height: int = 7
    max_steps: int = 200

    num_packages: int = 1
    battery_capacity: int = 50

    wall_fraction: float = 0.12 # ~12% of empty cells become walls

    reward_deliver: float = 20.0
    reward_pickup: float = 2.0
    penalty_step: float = -1.0
    penalty_bump: float = -5.0
    penalty_drop_wrong: float = -7.0

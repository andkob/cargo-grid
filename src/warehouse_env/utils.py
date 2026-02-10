from __future__ import annotations
import random
from dataclasses import dataclass

@dataclass
class RNG:
    """
    A deterministic random number generator wrapper used by the environment.

    This class isolates randomness from Python's global random state so that
    simulations are reproducible. Two environments created with the same seed
    will produce identical sequences of random values.

    Use from_seed() to construct instances instead of calling the constructor
    directly.
    """
    _r: random.Random

    @classmethod
    def from_seed(cls, seed: int | None) -> "RNG":
        if seed is None:
            seed = 0
        return cls(random.Random(seed))

    def randint(self, a: int, b: int) -> int:
        return self._r.randint(a, b)

    def choice(self, seq):
        return self._r.choice(seq)

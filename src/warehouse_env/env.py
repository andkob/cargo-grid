from __future__ import annotations
from typing import Any, Dict, Tuple
from .config import EnvConfig
from .state import EnvState, Package
from .utils import RNG
from .render import render_ansi

# Actions represented as integers:
# 0: up, 1: down, 2: left, 3: right, 4: pickup, 5: deliver
Action = int

# Observations are returned as a dictionary with mixed value types.
Obs = Dict[str, Any]

class WarehouseEnv:
    """
    A simple grid based warehouse environment.

    The agent moves on a 2D grid, can pick up a package when standing on it,
    and can deliver a carried package when standing on the goal cell.

    Public API
    reset(seed) -> initial observation
    step(action) -> (observation, reward, done, info)
    render() -> human readable string
    """

    def __init__(self, config: EnvConfig | None = None):
        """
        Create a new environment instance.

        Parameters
        config:
            Optional environment configuration. If None, uses EnvConfig()
            with its default values.

        Notes
        This does not start an episode. Call reset() before step().
        """
        self.cfg = config or EnvConfig()
        self._rng = RNG.from_seed(0)

        # No episode is active until reset() sets an EnvState
        self._state: EnvState | None = None

        # Goal is the bottom right cell of the grid
        self.goal_pos = (self.cfg.width - 1, self.cfg.height - 1)

    def reset(self, seed: int | None = None) -> Obs:
        """
        Start a new episode and return the initial observation.

        Parameters
        seed:
            Seed for deterministic randomness. Same seed produces the same
            initial package placement and any later random choices.

        Returns
        Obs:
            The initial observation dictionary.
        """
        # Reset RNG so the episode is reproducible from this seed
        self._rng = RNG.from_seed(seed)

        # Agent always starts in the top left cell
        agent_pos = (0, 0)

        # Spawn walls at random valid locations
        walls = self._spawn_walls(agent_pos)

        # Spawn packages at random valid locations
        packages = []
        for i in range(self.cfg.num_packages):
            packages.append(Package(id=i, pos=self._spawn_package_pos(agent_pos, walls)))

        # Create a fresh episode state
        self._state = EnvState(
            step_count=0,
            agent_pos=agent_pos,
            carrying_id=None,
            battery=self.cfg.battery_capacity,
            packages=packages,
            walls=walls,
        )

        # Return the observation for the starting state
        return self._obs(self._state)

    def step(self, action: Action) -> Tuple[Obs, float, bool, Dict[str, Any]]:
        """
        Apply an action and advance the simulation by one step.

        Actions
        0: move up
        1: move down
        2: move left
        3: move right
        4: pick up a package (if standing on one and not already carrying)
        5: drop a package (deliver if at goal while carrying)

        Parameters
        action:
            The chosen action as an integer.

        Returns
        observation:
            Observation after the step.
        reward:
            Step reward as a float.
        done:
            Whether the episode has terminated.
        info:
            Extra debug information such as events and done reason.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        s = self._state
        info: Dict[str, Any] = {"event": None}

        # Start with the per step penalty
        reward = 0.0
        reward += self.cfg.penalty_step

        bumped = False
        dropped_wrong = False
        picked_up = False
        delivered_now = False

        # Movement actions
        if action in (0, 1, 2, 3):
            # Compute the attempted next cell.
            nx, ny = self._move_target(s.agent_pos, action)

            # Apply movement only if still inside grid bounds
            if self._can_enter(nx, ny, s.walls):
                s.agent_pos = (nx, ny)
            else:
                # Out of bounds means agent stays in place
                bumped = True
        
        # Pickup action
        elif action == 4:
            # Can only pick up if not already carrying and standing on a package
            if (s.is_carrying):
                info["event"] = "pickup_failed_already_carrying"
            else:
                pid = self._package_id_at_agent(s)
                if pid is None:
                    info["event"] = "pickup_failed_no_package"
                else:
                    s.carrying_id = pid
                    # Invariant enforcement: attach immediately
                    self._set_package_pos(s, pid, s.agent_pos)
                    picked_up = True
                    info["event"] = "pickup"
        
        # Drop action.
        elif action == 5:
            # Drop anywhere. If at goal while carrying, counts as delivery
            if not s.is_carrying:
                info["event"] = "drop_failed_not_carrying"
            else:
                # should never happen, but being safe
                if s.carrying_id is None:
                    raise RuntimeError("Inconsistent state: carrying_id is None but is_carrying is True")
                
                pid = s.carrying_id
                self._set_package_pos(s, pid, s.agent_pos)

                if s.agent_pos == self.goal_pos:
                    self._mark_package_delivered(s, pid)
                    s.carrying_id = None
                    delivered_now = True
                    info["event"] = "deliver"
                else:
                    s.carrying_id = None
                    dropped_wrong = True
                    info["event"] = "drop"
        else:
            raise ValueError(f"Invalid action: {action}")

        # Apply bump penalty and annotate event
        if bumped:
            reward += self.cfg.penalty_bump
            info["event"] = "bump"

        # Apply drop penalty if wrong drop
        if dropped_wrong:
            reward += self.cfg.penalty_drop_wrong

        # Apply pickup reward
        if picked_up:
            reward += self.cfg.reward_pickup

        # Apply delivery reward
        if delivered_now:
            reward += self.cfg.reward_deliver

        # Keep carried package attached to the agent
        if s.is_carrying:
            # should never happen, but being safe
            if s.carrying_id is None:
                raise RuntimeError("Inconsistent state: carrying_id is None but is_carrying is True")
            self._set_package_pos(s, s.carrying_id, s.agent_pos)

        # Update time and resources each step
        s.step_count += 1
        s.battery = max(s.battery - 1, 0)

        # Termination checks
        done = False
        if s.step_count >= self.cfg.max_steps:
            done = True
            info["done_reason"] = "max_steps"
        elif s.battery == 0:
            done = True
            info["done_reason"] = "battery_empty"
        elif self._all_delivered(s):
            done = True
            info["done_reason"] = "all_delivered"

        # Store updated state and return the step tuple
        self._state = s
        return self._obs(s), float(reward), bool(done), info

    def render(self) -> str:
        """
        Render the current episode state as an ANSI friendly string.

        Returns an empty string if the environment has not been reset yet.
        """
        if self._state is None:
            return ""
        return render_ansi(self._state, self.cfg.width, self.cfg.height, self.goal_pos)

    def _obs(self, s: EnvState) -> Obs:
        """
        Convert internal EnvState into an external observation dictionary.
        """
        return {
            "agent_pos": s.agent_pos,
            "battery": s.battery,
            "carrying": s.carrying_id,
            "packages": [(p.pos, int(p.delivered)) for p in s.packages],
            "goal_pos": self.goal_pos,
            "walls": sorted(list(s.walls)),
            "step_count": s.step_count,
        }
    
    def _spawn_walls(self, agent_pos: tuple[int, int]) -> set[tuple[int, int]]:
        """
        Randomly generate wall positions on the grid.

        Walls will never be placed on the agent start position or the goal cell.
        The number of walls is determined by wall_fraction of available cells.
        """
        ax, ay = agent_pos
        gx, gy = self.goal_pos

        # Build the set of valid grid cells
        candidates: list[tuple[int, int]] = []
        for y in range(self.cfg.height):
            for x in range(self.cfg.width):
                if (x, y) == (ax, ay):
                    continue
                if (x, y) == (gx, gy):
                    continue
                candidates.append((x, y))

        # Randomly select a subset of candidates to become walls
        total = len(candidates)
        k = int(total * self.cfg.wall_fraction)

        # Pick uniformly from valid candidates
        walls: set[tuple[int, int]] = set()
        for _ in range(k):
            if not candidates:
                break
            pos = self._rng.choice(candidates)
            candidates.remove(pos)
            walls.add(pos)

        return walls

    def _spawn_package_pos(self, agent_pos: tuple[int, int], walls: set[tuple[int, int]]) -> tuple[int, int]:
        """
        Choose a random spawn position for a package.

        The position will never equal the agent start position or the goal cell.
        """
        ax, ay = agent_pos
        gx, gy = self.goal_pos

        # Build the set of valid grid cells
        candidates = []
        for y in range(self.cfg.height):
            for x in range(self.cfg.width):
                # Don't allow the agent start cell, goal cell, or wall cells
                pos = (x, y)
                if pos == (ax, ay):
                    continue
                if pos == (gx, gy):
                    continue
                if pos in walls:
                    continue
                candidates.append(pos)

        # Pick uniformly from valid candidates
        return self._rng.choice(candidates)

    def _move_target(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        """
        Compute the attempted next position for a movement action.
        """
        x, y = pos
        if action == 0:   # up
            return (x, y - 1)
        if action == 1:   # down
            return (x, y + 1)
        if action == 2:   # left
            return (x - 1, y)
        return (x + 1, y) # right (action == 3)

    def _can_enter(self, x: int, y: int, walls: set[tuple[int, int]]) -> bool:
        """
        Return True if (x, y) lies within the grid and is not a wall.
        """
        return 0 <= x < self.cfg.width and 0 <= y < self.cfg.height and (x, y) not in walls

    def _package_id_at_agent(self, s: EnvState) -> int | None:
        """
        Return Package id if the agent is standing on an undelivered package.
        """
        for p in s.packages:
            if (not p.delivered) and (p.pos == s.agent_pos):
                return p.id
        return None

    def _set_package_pos(self, s: EnvState, pid: int, new_pos: tuple[int, int]) -> None:
        """
        Move the package with the given id to a new position.

        This is used after a successful pickup action to move the package with
        the agent.
        """
        for p in s.packages:
            if p.id == pid:
                p.pos = new_pos
                return
        raise RuntimeError("Carrying package id not found in state")

    def _mark_package_delivered(self, s: EnvState, pid: int) -> None:
        """
        Mark the first undelivered package as delivered.

        This is used after a successful delivery action.
        """
        for p in s.packages:
            if p.id == pid:
                p.delivered = True
                return
        raise RuntimeError("Carrying package id not found in state")

    def _all_delivered(self, s: EnvState) -> bool:
        """
        Return True if all packages in the episode have been delivered.
        """
        for p in s.packages:
            if not p.delivered:
                return False
        return True
